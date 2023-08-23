import argparse
import os
import shutil
from glob import glob

import torch
from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.vnet_noise_RandomCrop import VNet
from networks.unet_3D import unet_3D
#from networks.vnet_auxDec_concat_DenseModi_decPointwise_concat import VNet
from test_3D_util import test_all_case



def net_factory(net_type="unet_3D", num_classes=3, in_channels=1):
    if net_type == "unet_3D":
        net = unet_3D(n_classes=num_classes, in_channels=in_channels).cuda()
    elif net_type == "unet_3D_dv_semi":
        net = unet_3D_dv_semi(n_classes=num_classes,
                              in_channels=in_channels).cuda()
    else:
        net = None
    return net

def Inference(FLAGS):
    #snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp, FLAGS.model)
    snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp, FLAGS.model)
    num_classes = 2
    #test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test,FLAGS.model)
    test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = VNet(n_channels= 1, n_classes=num_classes, n_filters=16, normalization='batchnorm', has_dropout=True).cuda()
    #net = net_factory(FLAGS.model, num_classes=2, in_channels=1)
    save_mode_path = os.path.join(
        snapshot_path, 'model2_iter_29000_dice_0.9006.pth')
    #save_mode_path = os.path.join(
    #    snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
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
                        default='LA/CTB_unetVnet_cropMse_SLboundary', help='experiment_name')
    parser.add_argument('--exp_test', type=str,
                        default='LA_test/CTB_unetVnet_cropMse_SLboundary', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='unet_vnet_3D', help='model_name')
    parser.add_argument('--gpu', type=str, default='6', help='GPU to use')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0,
                        help='apply NMS post-procssing?')
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    metric = Inference(FLAGS)
    print(metric)
