import os
import argparse
import torch
from networks.vnet_sdf import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/sohui/BraTS/data/BraTs2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='sassnet', help='model_name')
parser.add_argument('--model', type=str,  default='vnet_3D_96_32', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--max_iterations', type=int,  default=30000, help='maximum epoch number to train')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--labelnum', type=int,  default=50, help='random seed')
parser.add_argument('--total_labeled_num', type=int, default=250, help='total labeled data')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = '/data/sohui/BraTS/data/brats_2class_train_result/BraTs19_label_{}_{}/{}/{}_{}'\
        .format(FLAGS.labelnum, FLAGS.total_labeled_num,FLAGS.exp,FLAGS.model,FLAGS.max_iterations)

num_classes = 2

test_save_path = "/data/sohui/BraTS/data/brats_2class_test_result/BraTs19_label_{}_{}/{}/{}" \
        .format(FLAGS.labeled_num, FLAGS.total_labeled_num, FLAGS.exp, FLAGS.model)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

with open(FLAGS.root_path + '/test.txt', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/{}.h5".format(item.replace('\n', '').split(",")[0]) for item in image_list]


def test_calculate_metric(epoch_num):
    net = VNet(n_channels=1, n_classes=num_classes-1, normalization='batchnorm', has_dropout=False).cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_25600_dice_0.8183.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                           patch_size=(96,96,96), stride_xy=32, stride_z=32,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(FLAGS.iter)
    print(metric)

