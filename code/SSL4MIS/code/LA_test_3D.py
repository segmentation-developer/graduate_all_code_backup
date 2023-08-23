import argparse
import os
import shutil
from glob import glob
import numpy as np
import torch
from networks.vnet_tracoco_encKL import VNet
#from networks.unet_3D import unet_3D
#from networks.vnet_auxDec_concat_DenseModi_decPointwise_concat import VNet
from LA_test_3D_util import test_all_case
import torch.nn as nn

'''
def net_factory(net_type="unet_3D", num_classes=3, in_channels=1):
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
    #snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp, FLAGS.model)
    snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    #test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test,FLAGS.model)
    test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = VNet(n_channels= 1, n_classes=num_classes, n_filters=16, normalization='batchnorm', has_dropout=True).cuda()
    #net = nn.DataParallel(net).to(device)
    #net = net_factory(FLAGS.model, num_classes=2, in_channels=1)
    save_mode_path = os.path.join(
        snapshot_path, 'model2_iter_15000_dice_0.892.pth')
    #save_mode_path = os.path.join(
    #    snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.list", num_classes=num_classes,
                               patch_size=(112,112,80), stride_xy=18, stride_z=4, save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail,nms=FLAGS.nms)
    return avg_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/data/sohui/LA_dataset/2018LA_Seg_TrainingSet', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='LA/Vnet2TracocoKam_enc4KL_URUMpreds', help='experiment_name')
    parser.add_argument('--exp_test', type=str,
                        default='LA_test/Vnet2TracocoKam_enc4KL_URUMpreds', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='vnet_newdata_20000_2', help='model_name')
    parser.add_argument('--gpu', type=str, default='7', help='GPU to use')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0,
                        help='apply NMS post-procssing?')
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric, dice_list, hd_list = Inference(FLAGS, device=device)
    print(metric)
    print(np.mean(dice_list))
    print(np.std(dice_list))
    print(np.mean(hd_list))
    print(np.std(hd_list))
