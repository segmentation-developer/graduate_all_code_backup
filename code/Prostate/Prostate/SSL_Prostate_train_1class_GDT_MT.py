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
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import MSELoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataloaders import utils
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    Spacingd,
    RandRotate90d,
    CenterSpatialCropd,
    RandSpatialCropd,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    RandFlip,
    Spacing,
    RandRotate90,
    CenterSpatialCrop,
    RandSpatialCrop
)
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
#from networks.vnet_2task_SDM_2Dec import VNet
#from networks.vnet import VNet
from networks.vnet_2task_SDM_2Dec import VNet
from utils import losses, metrics, ramps
from MSDP_val_3D_GDT_MT import test_all_case
from utils.util import compute_sdf
import wandb
from skimage import segmentation as skimage_seg
#from mmcv import Config
import numpy
from itertools import cycle

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/Prostate/data/trim/ssl_data/centerCrop_200', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='UAMT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='test', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[208,208,128], ## brats_2class max_size : 148,188,156
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--class_name', type=int,  default=1)
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=40,
                    help='labeled data')
parser.add_argument('--total_labeled_num', type=int, default=290,
                    help='total labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default= 0.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str,  default='2,3', help='GPU to use')
parser.add_argument('--T', type=int,  default=8, help='ATD iter')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="SSL_Hprostate", config={}, reinit=True)
wandb.run.name = '{}/{}_{}_T={}'.format(args.exp,args.model,args.max_iterations,args.T)

if args.consistency_rampup == 0.0 :
    consistency_rampup = ((args.max_iterations//(args.total_labeled_num-args.labeled_num)) * 80 )/ 300
else :
    consistency_rampup = args.consistency_rampup

def get_current_consistency_weight(curr_epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(curr_epoch, consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data,alpha = 1 - alpha)


def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    labeled_bs = args.labeled_bs
    max_iterations = args.max_iterations
    num_classes = args.num_classes

    def create_model(ema=False):
        # Network definition
        model = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        #model = ConvNeXt(in_chans=1,drop_path_rate=0.4, layer_scale_init_value=1.0)
        #model = unet_3D( in_channels=1, n_classes=num_classes )
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    if len(args.gpu.split(',')) > 1:
        model = nn.DataParallel(model).to(device)
        ema_model = nn.DataParallel(ema_model).to(device)
    else :
        model = model.cuda()
        ema_model = ema_model.cuda()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),  # ALI
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8, 0.8, 0.8),
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(keys=['image', 'label'], roi_size=(208,208,128),random_size=False),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.20,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.30,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.40,
                max_k=3,
            ),

        ]
    )
    ul_train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAI"),  # ALI
            Spacingd(
                keys=["image"],
                pixdim=(0.8, 0.8, 0.8),
                mode=("bilinear"),              # mode가 sequence -> key 순서대로
            ),
            RandSpatialCropd(keys=['image'], roi_size=(208,208,128), random_size=False),
            RandFlipd(
                keys=["image"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[1],
                prob=0.20,
            ),
            RandFlipd(
                keys=["image"],
                spatial_axis=[2],
                prob=0.30,
            ),
            RandRotate90d(
                keys=["image"],
                prob=0.40,
                max_k=3,
            ),

        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8, 0.8, 0.8),
                mode=("bilinear", "nearest"),
            )
        ]
    )

    if args.class_name == 1:
        datasets = args.root_path + "/dataset.json"
        ul_datasets = args.root_path + "/dataset_unlabeled.json"
        print("total_prostate train : dataset.json")
    if args.class_name == 2:
        datasets = args.root_path + "/dataset_2.json"
        ul_datasets = args.root_path + "/dataset_unlabeled.json"
        print("transition zone train :dataset_2.json")
    train_files = load_decathlon_datalist(datasets, True, "training")
    ul_train_files = load_decathlon_datalist(ul_datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "test")

    ########## label train dataload
    db_train_SL = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )

    SL_trainloader = DataLoader(db_train_SL, batch_size=args.batch_size-args.labeled_bs, shuffle=False,
                                num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  ## 40개 안에서 shuffle

    ########## unlabel train dataload
    db_train_UL = CacheDataset(
        data=ul_train_files,
        transform=ul_train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )

    UL_trainloader = DataLoader(db_train_UL, batch_size=args.batch_size-args.labeled_bs, shuffle=False,
                                num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)  ## 40개 안에서 shuffle

    SL, UL = (cycle(SL_trainloader), UL_trainloader)

    ##########val dataload
    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001) #weight_decay=0.0005
    ce_loss = CrossEntropyLoss()        ##torch
    dice_loss = losses.DiceLoss(args.num_classes)       ##make
    mse_loss = MSELoss()        ##torch

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(UL_trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(UL_trainloader) + 1
    print('max_epoch:{}'.format(max_epoch))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(zip(SL,UL)):

            volume_batch, label_batch = sampled_batch[0]['image'].cuda(), sampled_batch[0]['label'].cuda()
            unlabeled_volume_batch = sampled_batch[1]['image'].cuda() ##unlabel
            volume_batch =  torch.cat((volume_batch,unlabeled_volume_batch),0)

            '''
            plt.figure(figsize=(18, 18))
            plt.subplot(6, 1, 1)
            plt.imshow(volume_batch[0][0][:,:,170:171].detach().cpu().numpy())
            plt.subplot(6, 1, 2)
            plt.imshow(volume_batch[1][0][:,:,170:171].detach().cpu().numpy())
            plt.subplot(6, 1, 3)
            plt.imshow(label_batch[0][0][:, :, 170:171].detach().cpu().numpy())
            plt.subplot(6, 1, 4)
            plt.imshow(label_batch[1][0][:, :,170:171].detach().cpu().numpy())
            plt.subplot(6, 1, 5)
            plt.imshow(unlabeled_volume_batch[0][0][:, :, 100:101].detach().cpu().numpy())
            plt.subplot(6, 1, 6)
            plt.imshow(unlabeled_volume_batch[1][0][:, :, 100:101].detach().cpu().numpy())
            plt.tight_layout()
            plt.show()
            '''

            outputs_2class, outputs_1class_tanh = model(volume_batch)  ## model = student
            outputs_soft_2class = torch.softmax(outputs_2class, dim=1)

            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:].cpu(
                ).numpy(), outputs_1class_tanh[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()


            T = args.T
            _, _, d, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, d, w, h]).cuda()
            SDM_preds = torch.zeros([stride * T, 1, d, w, h]).cuda()
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)], SDM_preds[
                                                                2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)  ## unlabel의 shape를 *2로 복사한 값에 새로운 noise를 >계속 넣어줘서 teacher model에 계속 실험  -> preds에 계속 결과 쌓기
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, d, w, h)  # 8,bs,ch,w,h,d
            seg_avg = torch.mean(preds, dim=0)  # 2,2,w,h,d

            SDM_preds = SDM_preds.reshape(T, stride, 1, d, w, h)
            SDM_avg = torch.mean(SDM_preds, dim=0)

############ supervised loss
            loss_ce = ce_loss(outputs_2class[:labeled_bs], label_batch.squeeze(1).long())
            loss_dice = dice_loss(outputs_soft_2class[:labeled_bs], label_batch[:labeled_bs].float())
            loss_sdf = mse_loss(outputs_1class_tanh[:labeled_bs, 0, ...], gt_dis)

            supervised_loss = 0.5 * (loss_dice + loss_ce+ loss_sdf)      ## (추측)supervised : SDM의 영향 > segmentation 영향 -> SDM loss 적게 반영

############ unsupervised loss
            #consistency_weight = get_current_consistency_weight(iter_num* (args.consistency_rampup/(args.max_iterations)) )
            #consistency_weight = get_current_consistency_weight(iter_num // (args.max_iterations / args.consistency_rampup))
            consistency_weight = get_current_consistency_weight(iter_num//len(UL_trainloader))
            consistency_mse = torch.mean((outputs_soft_2class[labeled_bs:] - seg_avg) ** 2)  # (batch, 2, 112,112,80)

            consistency_1class_loss = mse_loss(outputs_1class_tanh[labeled_bs:], SDM_avg)

            loss = supervised_loss + consistency_weight * (consistency_1class_loss +consistency_mse)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            #writer.add_scalar('info/consistency_loss',
            #                  consistency_mse, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "loss_ce": loss_ce.item(),
                "loss_dice": loss_dice.item(),
                "loss_sdf_1class": loss_sdf.item(),
                "consistency_segMse_loss" : consistency_mse.item(),
                "consistency_1class_loss": consistency_1class_loss.item(),
                "consistency_weight":consistency_weight


            })


            if iter_num > 0 and iter_num % 1000 == 0:
                model.eval()
                avg_metric= test_all_case(
                    model,val_loader, num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))


                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.class_name == 1:
        snapshot_path = "/data/sohui/Prostate/prostate_1c_train_result/{}/{}_{}".format(args.exp, args.model,
                                                                                        args.max_iterations)
    elif args.class_name == 2:
        snapshot_path = "/data/sohui/Prostate/TZ_1c_train_result/{}/{}_{}".format(args.exp, args.model,
                                                                                  args.max_iterations)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('..', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)