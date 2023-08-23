# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the train_validation_test set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the train_validation_test set and test set to report the results.
#   We found that the random data split has some bias (the train_validation_test set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained model to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import MSELoss
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.utils import make_grid

#from config import get_config
from dataloaders.brats.brats2019 import TwoStreamBatchSampler
from networks.unet_3D import unet_3D
from networks.vnet_tracoco_dsv import VNet
from utils import losses, ramps
from HV_val_3D import test_all_case
from skimage import segmentation as skimage_seg
import matplotlib.pyplot as plt
import torch.nn.functional as F

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    SpatialPadd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from itertools import cycle
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/HV/Task08_HepaticVessel', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='HV_result/Unet2_dsv1URUM', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_newdata_20000_1', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[144,144,64],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
#parser.add_argument(
#    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=24,
                    help='labeled data')
parser.add_argument('--total_labeled_num', type=int, default=242,
                    help='total labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str,  default='5,6', help='GPU to use')
parser.add_argument('--add', type=float,  default=1e-8)
parser.add_argument('--fold', type=str,  default=None)
parser.add_argument('--temperature', type=float,  default=0.07)
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
args = parser.parse_args()
#config = get_config(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="SSL_Brats", config={}, reinit=True)
wandb.run.name = '{}/{}'.format(args.exp,args.model)

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model



def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


'''
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in (ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
'''

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = unet_3D(n_classes=num_classes, in_channels=1)
        #model = VNet(n_channels=1, n_classes=num_classes, n_filters=16, normalization='batchnorm',has_dropout=True).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model1 = nn.DataParallel(model1).to(device)
    model2 = unet_3D(n_classes=num_classes, in_channels=1)
    model2 = nn.DataParallel(model2).to(device)
    model1.train()
    model2.train()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def class_create(input_tensor,n_classes):
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def compute_bound(img_gt):  # label_batch[:labeled_bs, 0, ...].shape
        """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM)
        sdf(x) = 0; x in segmentation boundary
                 -inf|x-y|; x in segmentation
                 +inf|x-y|; x out of segmentation
        normalize sdf to [-1,1]
        """

        img_gt = img_gt
        boundary = np.zeros(img_gt.shape)

        for b in range(img_gt.shape[0]):  # batch size=2
            posmask = img_gt[b].astype(bool)
            boundary[b] = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)

        return boundary

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="LPI"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.25, 1.25, 5.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-1024,
                a_max= 2000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="label"),
            #SpatialPadd(keys=["image", "label"], spatial_size=(192, 192, 48), mode="constant"),

        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.25,1.25,5.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
        ]
    )

    datasets = args.root_path + "/dataset.json"
    datalist = load_decathlon_datalist(datasets, True, "training")      #302개
    train_files, val_files = datalist[:-60], datalist[-60:]   # 242,60
    SL_train_files = train_files[:args.labeled_num]
    UL_train_files = train_files[args.labeled_num:]

    ##########train dataload
    db_train_SL = CacheDataset(
        data=SL_train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    db_train_UL = CacheDataset(
        data=UL_train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )

    SL_trainloader = DataLoader(db_train_SL, batch_size=1,shuffle=False,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    UL_trainloader = DataLoader(db_train_UL, batch_size=1,shuffle=False,
                                num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    SL,UL = (cycle(SL_trainloader),UL_trainloader)      #iter(zip()) del

    ##########val dataload
    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )


    iter_num = 0
    max_epoch = max_iterations // len(UL) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    nt_xent_criterion_ch = losses.NTXentLoss( temperature= 0.5, use_cosine_similarity= True)
    lr_ = base_lr
    for epoch_num in range(1):
        for i_batch, sampled_batch in enumerate(zip(SL,UL)):       #0,1 : SL_trainloader(bs:1), UL_trainloader(bs:1)
            sl_volume_batch, sl_label_batch = sampled_batch[0]['image'].squeeze().cuda(), sampled_batch[0]['label'].squeeze().cuda()
            ul_volume_batch, ul_label_batch = sampled_batch[1]['image'].squeeze().cuda(), sampled_batch[1]['label'].squeeze().cuda()

            print("SL_img:{}".format( sl_volume_batch.size()))

            print("UL_img:{}".format( ul_volume_batch.size()))


if __name__ == "__main__":

    snapshot_path = "/data/sohui/HV/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
