import os
import argparse
import torch
import torch.nn as nn
#from networks.vnet_sdf import VNet
from test_util_DTC import test_all_case
#from networks.vnet_sdf import VNet
#from networks.attentionNet import AttU_Net
#from networks.voxResNet import VoxResNet
#from networks.attention_Unet3D import Attention_UNet
#from networks.vnet_attention_sdf import Vattention_Unet
from networks.vnet_auxDec_concat_DenseModi_decPointwise_concat import VNet


print(torch.cuda.is_available())
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/LA_dataset/2018LA_Seg_TestingSet', help='Name of Experiment')
parser.add_argument('--model', type=str,
                    default='vnet_auxDec_concat_DenseM_decP_concat_addconv', help='model_name')
parser.add_argument('--exp', type=str,
                    default='LA_test', help='experiment_name')
parser.add_argument('--gpu', type=str,  default='5', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1,
                    help='apply NMS post-procssing?')


FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
snapshot_path = "/data/sohui/LA_dataset/LA/{}".format(FLAGS.model)          #test dataset불러오는 곳
#snapshot_path = "/home/sohui/DTC/model/{}".format(FLAGS.model)

num_classes = 2

test_save_path = "/data/sohui/LA_dataset/LA_test/{}".format(FLAGS.model)
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test_list.txt', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
              image_list]


def test_calculate_metric():
    net = VNet().cuda()


    save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_mode_path))

    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes-1,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=True, test_save_path=test_save_path,
                               metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()  # 6000
    print(metric)
