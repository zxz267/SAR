import torch

class Config:
    pre = 'SAR'
    dataset = 'FreiHAND'
    output_root = './output'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # network
    backbone = 'resnet34'
    num_stage = 2
    num_FMs = 8
    feature_size = 64
    heatmap_size = 32
    num_vert = 778
    num_joint = 21
    # training
    batch_size = 64
    lr = 3e-4
    total_epoch = 50
    input_img_shape = (256, 256)
    depth_box = 0.3
    num_worker = 16
    # -------------
    save_epoch = 1
    eval_interval = 1
    print_iter = 10
    num_epoch_to_eval = 80
    # -------------
    checkpoint = ''  # put the path of the trained model's weights here
    continue_train = False
    vis = False
    # -------------
    experiment_name = pre + '_{}'.format(backbone) + '_Stage{}'.format(num_stage) + '_Batch{}'.format(batch_size) + \
                      '_lr{}'.format(lr) + '_Size{}'.format(input_img_shape[0]) + '_Epochs{}'.format(total_epoch)

cfg = Config()
