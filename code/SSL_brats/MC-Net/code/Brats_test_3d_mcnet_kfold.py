import os
import argparse
import torch

from networks.net_factory import net_factory
from utils.test_patch_kfold import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='Brats19_kfold', help='dataset_name')
parser.add_argument('--root_path', type=str, default='/data/sohui/BraTS/data/BraTs2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MCNet+_kfold/mcnet3d_v2_30000_fold5', help='exp_name')
parser.add_argument('--model', type=str,  default='mcnet3d_v2', help='model_name')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--labelnum', type=int, default=56, help='labeled data')
parser.add_argument('--max_samples', type=int,  default=268, help='maximum samples to train')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')
parser.add_argument('--max_iterations', type=int,  default=30000, help='maximum iteration to train')
parser.add_argument('--fold', type=int,  default=5, help='k fold cross validation')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = '/data/sohui/BraTS/data/brats_2class_train_result/BraTs19_label_{}_{}/{}' \
    .format(FLAGS.labelnum, FLAGS.max_samples, FLAGS.exp)
test_save_path = "/data/sohui/BraTS/data/brats_2class_test_result/BraTs19_label_{}_{}/{}" \
    .format(FLAGS.labelnum, FLAGS.max_samples, FLAGS.exp)

num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = FLAGS.root_path + 'data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = FLAGS.root_path + 'data/Pancreas'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

elif FLAGS.dataset_name == "Brats19":
     patch_size = (96, 96, 96)
     with open(FLAGS.root_path + '/{}'.format('test.txt'), 'r') as f:
         image_list = f.readlines()
     image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + ".h5" for item in image_list]

elif FLAGS.dataset_name == "Brats19_kfold":
     patch_size = (96, 96, 96)
     with open(FLAGS.root_path + '/fold{}/test.txt'.format(FLAGS.fold), 'r') as f:
        image_list = f.readlines()
     image_list = [FLAGS.root_path + "/" + item.replace('\n', '') for item in image_list]
     print('/fold{}/test.txt'.format(FLAGS.fold))

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)

def test_calculate_metric():
    
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    save_mode_path = os.path.join(snapshot_path, 'iter_19800_dice_0.80085199785188.pth')
    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas_CT":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Brats19":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(96, 96, 96), stride_xy=32, stride_z=32,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Brats19_kfold":
        avg_metric = test_all_case(FLAGS.model, 1, net, image_list, num_classes=num_classes,
                        patch_size=(96, 96, 96), stride_xy=32, stride_z=32,
                        save_result=True, test_save_path=test_save_path,
                        metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)
