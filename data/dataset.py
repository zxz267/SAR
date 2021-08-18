from data.processing import xyz2uvd, uvd2xyz, read_img, load_db_annotation, augmentation, cv2pil
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as standard
from config import cfg
import json
import os
from data.processing import get_focal_pp
from utils.visualize import render_mesh_multi_views, draw_2d_skeleton
from utils.mano import MANO
import cv2

class FreiHAND(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode
        assert self.mode in ['training', 'evaluation'], 'mode error'
        # load annotations
        self.anno_all = load_db_annotation(root, self.mode)
        if self.mode == 'evaluation':
            root = os.path.join(root, 'bbox_root_freihand_output.json')
            self.root_result = []
            with open(root) as f:
                annot = json.load(f)
            for i in range(len(annot)):
                self.root_result.append(np.array(annot[i]['root_cam']))
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.transform = standard.Compose([standard.ToTensor(), standard.Normalize(*mean_std)])
        self.versions = ['gs', 'hom', 'sample', 'auto']

        # mano
        self.mano = MANO()
        self.face = self.mano.face

    def __getitem__(self, idx):
        if self.mode == 'training':
            version = self.versions[idx // len(self.anno_all)]
        else:
            version = 'gs'
        idx = idx % len(self.anno_all)
        img = read_img(idx, self.root, self.mode, version)
        bbox_size = 130
        bbox = [112 - bbox_size//2, 112 - bbox_size//2, bbox_size, bbox_size]
        img, img2bb_trans, bb2img_trans, _, _,  = \
            augmentation(img, bbox, self.mode, exclude_flip=True)
        img = cv2pil(img)
        img = self.transform(img)
        if self.mode == 'training':
            K, mesh_xyz, pose_xyz, scale = self.anno_all[idx]
            K, mesh_xyz, pose_xyz, scale = [np.array(x) for x in [K, mesh_xyz, pose_xyz, scale]]
            # concat mesh and pose label
            all_xyz = np.concatenate((mesh_xyz, pose_xyz), axis=0)
            all_uvd = xyz2uvd(all_xyz, K)
            # affine transform x,y coordinates
            uv1 = np.concatenate((all_uvd[:, :2], np.ones_like(all_uvd[:, :1])), 1)
            all_uvd[:, :2] = np.dot(img2bb_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
            # wrist is the relative joint
            root_depth = all_uvd[cfg.num_vert:cfg.num_vert+1, 2:3].copy()
            all_uvd[:, 2:3] = (all_uvd[:, 2:3] - root_depth)
            # box to normalize depth
            all_uvd[:, 2:3] /= cfg.depth_box
            # img normalize
            all_uvd[:, :2] = all_uvd[:, :2] / (cfg.input_img_shape[0] // 2) - 1
            inputs = {'img': np.float32(img)}
            targets = {'mesh_pose_uvd': np.float32(all_uvd)}
            meta_info = {}
        else:
            K, scale = self.anno_all[idx]
            K, scale = [np.array(x) for x in [K, scale]]
            inputs = {'img': np.float32(img)}
            targets = {}
            meta_info = {
                'img2bb_trans': np.float32(img2bb_trans),
                'bb2img_trans': np.float32(bb2img_trans),
                'root_depth': np.float32(self.root_result[idx][2][None]),
                'K': np.float32(K),
                'scale': np.float32(scale)}
        return inputs, targets, meta_info

    def __len__(self):
        if self.mode == 'training':
            return len(self.anno_all) * 4
        else:
            return len(self.anno_all)

    def evaluate(self, outs, meta_info, cur_sample_idx):
        coords_uvd = outs['coords']
        batch = coords_uvd.shape[0]
        eval_result = {'pose_out': list(), 'mesh_out': list()}
        for i in range(batch):
            coord_uvd_crop, root_depth, img2bb_trans, bb2img_trans, K, scale = \
                coords_uvd[i], meta_info['root_depth'][i], meta_info['img2bb_trans'][i], \
                meta_info['bb2img_trans'][i], meta_info['K'][i], meta_info['scale'][i]
            coord_uvd_crop[:, 2] = coord_uvd_crop[:, 2] * cfg.depth_box + root_depth
            coord_uvd_crop[:, :2] = (coord_uvd_crop[:, :2] + 1) * cfg.input_img_shape[0] // 2
            # back to original image
            coord_uvd_full = coord_uvd_crop.copy()
            uv1 = np.concatenate((coord_uvd_full[:, :2], np.ones_like(coord_uvd_full[:, :1])), 1)
            coord_uvd_full[:, :2] = np.dot(bb2img_trans, uv1.transpose(1, 0)).transpose(1, 0)[:, :2]
            coord_xyz = uvd2xyz(coord_uvd_full, K)
            pose_xyz = coord_xyz[cfg.num_vert:]
            mesh_xyz = coord_xyz[:cfg.num_vert]
            eval_result['pose_out'].append(pose_xyz.tolist())
            eval_result['mesh_out'].append(mesh_xyz.tolist())
            if cfg.vis:
                mesh_xyz_crop = uvd2xyz(coord_uvd_crop[:cfg.num_vert], K)
                vis_root = os.path.join(cfg.output_root, 'FreiHAND_vis')
                if not os.path.exists(vis_root):
                    os.makedirs(vis_root)
                idx = cur_sample_idx + i
                img_full = read_img(idx, self.root, 'evaluation', 'gs')
                img_crop = cv2.warpAffine(img_full, img2bb_trans, cfg.input_img_shape, flags=cv2.INTER_LINEAR)
                focal, pp = get_focal_pp(K)
                cam_param = {'focal': focal, 'princpt': pp}
                img_mesh, view_1, view_2 = render_mesh_multi_views(img_crop, mesh_xyz_crop, self.face, cam_param)
                path_mesh_img = os.path.join(vis_root, 'render_mesh_img_{}.png'.format(idx))
                cv2.imwrite(path_mesh_img, img_mesh)
                path_mesh_view1 = os.path.join(vis_root, 'render_mesh_view1_{}.png'.format(idx))
                cv2.imwrite(path_mesh_view1, view_1)
                path_mesh_view2 = os.path.join(vis_root, 'render_mesh_view2_{}.png'.format(idx))
                cv2.imwrite(path_mesh_view2, view_2)
                path_joint = os.path.join(vis_root, 'joint_{}.png'.format(idx))
                vis = draw_2d_skeleton(img_crop, coord_uvd_crop[cfg.num_vert:])
                cv2.imwrite(path_joint, vis)
                path_img = os.path.join(vis_root, 'img_{}.png'.format(idx))
                cv2.imwrite(path_img, img_crop)

        return eval_result

    def print_eval_result(self, eval_result):
        output_json_save_path = os.path.join(cfg.output_root, 'pred.json')
        with open(output_json_save_path, 'w') as fo:
            json.dump([eval_result['pose_out'], eval_result['mesh_out']], fo)
        print('Dumped %d joints and %d verts predictions to %s'
              % (len(eval_result['pose_out']), len(eval_result['mesh_out']), output_json_save_path))

def get_dataset(dataset, mode):
    if dataset == 'FreiHAND':
        return FreiHAND(os.path.join('./data', dataset), mode)

