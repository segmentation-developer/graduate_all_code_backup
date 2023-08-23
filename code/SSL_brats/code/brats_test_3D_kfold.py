import argparse
import os
import shutil

import torch
#from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.convNext_3D_unet import ConvNeXt
#from networks.unet_3D import unet_3D
#from networks.vnet_auxDec_concat_DenseModi_decPointwise_concat import VNet
#from networks.vnet import VNet
from networks.vnet_2task_SDM_2Dec import VNet
from brats_test_3D_util import test_all_case
import torch.nn as nn


def Inference(FLAGS, device):
    #snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp, FLAGS.model)
    snapshot_path = '/data/sohui/BraTS/data/brats_2class_train_result/BraTs19_label_{}_{}/{}/{}' \
        .format(FLAGS.labeled_num, FLAGS.total_data_num, FLAGS.exp, FLAGS.model)
    num_classes = 2
    #test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test,FLAGS.model)
    test_save_path = "/data/sohui/BraTS/data/brats_2class_test_result/BraTs19_label_{}_{}/{}/{}" \
        .format(FLAGS.labeled_num, FLAGS.total_data_num, FLAGS.exp, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True).cuda()
    if len(FLAGS.gpu.split(',')) > 1:
        net = nn.DataParallel(net).to(device)
    else:
        net = net.cuda()
    #net = net_factory(FLAGS.model, num_classes=2, in_channels=1)
    save_mode_path = os.path.join(
        snapshot_path, 'iter_13400_dice_0.8888.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric, avg_uncertainty = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96,96,96), stride_xy=32, stride_z=32, save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail,nms=FLAGS.nms, fold=FLAGS.fold)
    return avg_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/data/sohui/BraTS/data/BraTs2019', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='kfold/GDT-MT_opti', help='experiment_name')
    parser.add_argument('--model', type=str, default='vnet_3D_96_32_30000_T=6_fold4_consisW1.0', help='model_name')
    parser.add_argument('--gpu', type=str, default='4', help='GPU to use')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0,
                        help='apply NMS post-procssing?')
    parser.add_argument('--labeled_num', type=int, default=56,
                        help='labeled data')
    parser.add_argument('--total_data_num', type=int, default=268,
                        help='total labeled data')
    parser.add_argument('--fold', type=int, default=4, help='k fold cross validation')

    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = Inference(FLAGS, device=device)
    print(metric)
