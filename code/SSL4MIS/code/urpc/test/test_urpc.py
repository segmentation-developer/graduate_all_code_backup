import argparse
import os
import shutil

import torch

from networks.vnet_concat_dv_semi_2Dec import VNet
from code.urpc.test_urpc_util import test_all_case
import torch.nn as nn
'''
def net_factory(net_type="unet_3D", num_classes=2, in_channels=1):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=num_classes, in_channels=in_channels).cuda()
    elif net_type == "unet_3D_dv_semi":
        net = unet_3D_dv_semi(n_classes=num_classes,
                              in_channels=in_channels).cuda()
    else:
        net = None
    return net
'''

def Inference(FLAGS,device):
    snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test,FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    #net = net_factory(FLAGS.model, num_classes, in_channels=1)
    net = VNet(n_channels= 1, n_classes=num_classes, n_filters=16, normalization='batchnorm', has_dropout=True).cuda()
    net = nn.DataParallel(net).to(device)
    #paths = snapshot_path+'/iter*'
    save_mode_path = os.path.join(snapshot_path, FLAGS.fold)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test_list.txt", num_classes=num_classes,
                               patch_size=(112,112,80), stride_xy=18, stride_z=4, save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail,nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/data/sohui/LA_dataset/2018LA_Seg_TestingSet', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='LA/URPC_kfold_vnetConcat', help='experiment_name')
    parser.add_argument('--exp_test', type=str,
                        default='LA_test/URPC_kfold_vnetConcat/0_nms_test', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default="vnet", help='model_name')
    parser.add_argument('--path', type=str, default='None')
    parser.add_argument('--gpu', type=str, default='6', help='GPU to use')
    parser.add_argument('--fold', type=str, default='fold_0_iter_8000_dice_0.8912.pth')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=1,
                        help='apply NMS post-procssing?')

    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric = Inference(FLAGS,device=device)
    print(metric)
