import argparse
import os
import shutil

import torch
#from networks.unet_3D_dv_semi import unet_3D_dv_semi
from networks.convNext_3D_unet import ConvNeXt
#from networks.unet_3D import unet_3D
#from networks.vnet_auxDec_concat_DenseModi_decPointwise_concat import VNet
from networks.vnet_2task_SDM_2Dec import VNet
from brats_test_3D_util_SDM import test_all_case
import torch.nn as nn


def Inference(FLAGS, device):
    #snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp, FLAGS.model)
    snapshot_path = '/data/sohui/BraTS/data/brats_2class_train_result/BraTs19_label_{}_{}/{}/{}' \
        .format(FLAGS.labeled_num, FLAGS.total_labeled_num, FLAGS.exp, FLAGS.model)
    num_classes = 2
    #test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test,FLAGS.model)
    test_save_path = "/data/sohui/BraTS/data/brats_2class_test_result/BraTs19_label_{}_{}/{}_gtDisFore_{}_result/{}" \
        .format(FLAGS.labeled_num, FLAGS.total_labeled_num, FLAGS.exp,FLAGS.gt_dis_fore, FLAGS.model)
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
        snapshot_path, 'iter_7800_dice_0.8554.pth')
    #save_mode_path = os.path.join(
    #    snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    avg_metric = test_all_case(net, base_dir=FLAGS.root_path, method=FLAGS.model, test_list="test.txt", num_classes=num_classes,
                               patch_size=(96,96,96), stride_xy=32, stride_z=32, save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail,nms=FLAGS.nms, gt_dis_fore=FLAGS.gt_dis_fore)
    return avg_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/data/sohui/BraTS/data/BraTs2019', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='MT_UA2avg_2task_SDM_2Dec', help='experiment_name')
    parser.add_argument('--exp_test', type=str,
                        default='MT_UA2avg_2task_SDM_2Dec', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='vnet_3D_96--max_iterations=30000_30000', help='model_name')
    parser.add_argument('--gpu', type=str, default='5', help='GPU to use')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0,
                        help='apply NMS post-procssing?')
    parser.add_argument('--labeled_num', type=int, default=25,
                        help='labeled data')
    parser.add_argument('--total_labeled_num', type=int, default=250,
                        help='total labeled data')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--gt_dis_fore', type=str,
                        default=False, help='SDM range')
    FLAGS = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = Inference(FLAGS, device=device)
    print(metric)
