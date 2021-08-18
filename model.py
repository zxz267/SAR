import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from utils.mano import MANO
from config import cfg
from loss import EdgeLengthLoss, NormalVectorLoss


class SoftHeatmap(nn.Module):
    def __init__(self, size, kp_num):
        super(SoftHeatmap, self).__init__()
        self.size = size
        self.beta = nn.Conv2d(kp_num, kp_num, 1, 1, 0, groups=kp_num, bias=False)
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1])
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size])
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)

    def forward(self, x):
        s = list(x.size())
        scoremap = self.beta(x)
        scoremap = scoremap.view([s[0], s[1], s[2] * s[3]])
        scoremap = F.softmax(scoremap, dim=2)
        scoremap = scoremap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = scoremap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2)
        scoremap_y = scoremap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2)
        keypoint_uv = torch.stack([soft_argmax_x, soft_argmax_y], dim=2)
        return keypoint_uv, scoremap

class GraphConv(nn.Module):
    def __init__(self, num_joint, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.adj = nn.Parameter(torch.eye(num_joint).float().to(cfg.device), requires_grad=True)

    def laplacian(self, A_hat):
        D_hat = torch.sum(A_hat, 1, keepdim=True) + 1e-5
        L = 1 / D_hat * A_hat
        return L

    def forward(self, x):
        batch = x.size(0)
        A_hat = self.laplacian(self.adj)
        A_hat = A_hat.unsqueeze(0).repeat(batch, 1, 1)
        out = self.fc(torch.matmul(A_hat, x))
        return out

class SAIGB(nn.Module):
    def __init__(self, backbone_channels, num_FMs, feature_size, num_vert, template):
        super(SAIGB, self).__init__()
        self.template = torch.Tensor(template).to(cfg.device)  # self.mano.template
        self.backbone_channels = backbone_channels
        self.feature_size = feature_size
        self.num_vert = num_vert
        self.num_FMs = num_FMs
        self.group = nn.Sequential(
            nn.Conv2d(self.backbone_channels, self.num_FMs * self.num_vert, 1),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        feature = self.group(x).view(-1, self.num_vert, self.feature_size * self.num_FMs)
        template = self.template.repeat(x.shape[0], 1, 1)
        init_graph = torch.cat((feature, template), dim=2)
        return init_graph

class GBBMR(nn.Module):
    def __init__(self, in_dim, num_vert, num_joint, heatmap_size):
        super(GBBMR, self).__init__()
        self.in_dim = in_dim
        self.num_vert = num_vert
        self.num_joint = num_joint
        self.num_total = num_vert + num_joint
        self.heatmap_size = heatmap_size
        self.soft_heatmap = SoftHeatmap(self.heatmap_size, self.num_total)
        self.reg_xy = nn.Sequential(
            GraphConv(self.num_vert, self.in_dim, self.heatmap_size ** 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            GraphConv(self.num_vert, self.heatmap_size ** 2, self.heatmap_size ** 2),
        )
        self.reg_z = nn.Sequential(
            GraphConv(self.num_vert, self.in_dim, self.heatmap_size ** 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            GraphConv(self.num_vert, self.heatmap_size ** 2, self.heatmap_size ** 2),
        )
        self.mesh2pose_hm = nn.Linear(self.num_vert, self.num_joint)
        self.mesh2pose_dm = nn.Linear(self.num_vert, self.num_joint)

    def forward(self, x):
        init_graph = x
        heatmap_xy_mesh = self.reg_xy(init_graph).view(-1, self.num_vert, self.heatmap_size, self.heatmap_size)
        heatmap_z_mesh = self.reg_z(init_graph).view(-1, self.num_vert, self.heatmap_size, self.heatmap_size)
        heatmap_xy_joint = self.mesh2pose_hm(heatmap_xy_mesh.transpose(1, 3)).transpose(1, 3)
        heatmap_z_joint = self.mesh2pose_dm(heatmap_z_mesh.transpose(1, 3)).transpose(1, 3)
        heatmap_xy = torch.cat((heatmap_xy_mesh, heatmap_xy_joint), dim=1)
        heatmap_z = torch.cat((heatmap_z_mesh, heatmap_z_joint), dim=1)
        coord_xy, latent_heatmaps = self.soft_heatmap(heatmap_xy)
        depth_maps = latent_heatmaps * heatmap_z
        coord_z = torch.sum(
            depth_maps.view(-1, self.num_total, depth_maps.shape[2] * depth_maps.shape[3]), dim=2, keepdim=True)
        joint_coord = torch.cat((coord_xy, coord_z), 2)
        joint_coord[:, :, :2] = joint_coord[:, :, :2] / (self.heatmap_size // 2) - 1
        return joint_coord, latent_heatmaps, depth_maps

class SAR(nn.Module):
    def __init__(self):
        super(SAR, self).__init__()
        mano = MANO()
        backbone = models.__dict__[cfg.backbone](pretrained=True)
        self.extract_mid = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu,
                                         backbone.maxpool, backbone.layer1, backbone.layer2)
        self.extract_high = []
        self.saigb = []
        self.gbbmr = []
        self.fuse = []
        for i in range(cfg.num_stage):
            backbone = models.__dict__[cfg.backbone](pretrained=True)
            channel = backbone.fc.in_features
            self.extract_high.append(nn.Sequential(backbone.layer3, backbone.layer4))
            self.saigb.append(SAIGB(channel, cfg.num_FMs, cfg.feature_size, cfg.num_vert, mano.template))
            self.gbbmr.append(GBBMR(cfg.num_FMs*cfg.feature_size+3, cfg.num_vert, cfg.num_joint, cfg.heatmap_size))
            if i > 0:
                self.fuse.append(nn.Conv2d(channel // 4 + cfg.num_joint * 2, channel // 4, 1))
        self.extract_high = nn.ModuleList(self.extract_high)
        self.saigb = nn.ModuleList(self.saigb)
        self.gbbmr = nn.ModuleList(self.gbbmr)
        self.fuse = nn.ModuleList(self.fuse)

        self.coord_loss = nn.L1Loss()
        self.normal_loss = NormalVectorLoss(mano.face)
        self.edge_loss = EdgeLengthLoss(mano.face)

    def forward(self, x, target=None):
        x = x['img'].to(cfg.device)
        outs = {'coords': []}
        lhms = []
        dms = []
        feat_mid = self.extract_mid(x)
        for i in range(cfg.num_stage):
            if i > 0:
                feat = self.fuse[i-1](
                    torch.cat((feat_mid,
                               lhms[i-1][:, cfg.num_vert:],
                               dms[i-1][:, cfg.num_vert:]), dim=1))
            else:
                feat = feat_mid
            feat_high = self.extract_high[i](feat)
            init_graph = self.saigb[i](feat_high)
            coord, lhm, dm = self.gbbmr[i](init_graph)
            outs['coords'].append(coord)
            lhms.append(lhm)
            dms.append(dm)
        if self.training:
            loss = {}
            mesh_pose_uvd = target['mesh_pose_uvd'].to(cfg.device)
            for i in range(cfg.num_stage):
                loss['coord_{}'.format(i)] = self.coord_loss(outs['coords'][i], mesh_pose_uvd)
                loss['normal_{}'.format(i)] = \
                    self.normal_loss(outs['coords'][i][:, :cfg.num_vert], mesh_pose_uvd[:, :cfg.num_vert]).mean() * 0.1
                loss['edge_{}'.format(i)] = \
                    self.edge_loss(outs['coords'][i][:, :cfg.num_vert], mesh_pose_uvd[:, :cfg.num_vert]).mean()
            return loss
        else:
            outs['coords'] = outs['coords'][-1]
            return outs

def get_model():
    return SAR()

if __name__ == '__main__':
    import torch
    input = torch.rand(2, 3, 256, 256)
    net = SAR()
    output = net(input)
    print(output)



