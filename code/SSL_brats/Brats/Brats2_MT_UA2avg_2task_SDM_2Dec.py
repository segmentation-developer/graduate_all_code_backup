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
from dataloaders.brats.brats2019 import (BraTS2019, RandomCrop,CenterCrop,
                                         RandomRotFlip, ToTensor,
                                         TwoStreamBatchSampler)
from networks.unet_3D import unet_3D
from networks.vnet_2task_SDM_2Dec import VNet
from networks.convNext_3D_unet import ConvNeXt
from utils import losses, metrics, ramps
from brats_val_3D_2task import test_all_case
from utils.util import compute_sdf
import wandb
from skimage import segmentation as skimage_seg
#from mmcv import Config



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/BraTS/data/BraTs2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='UAMT', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='test', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96,96,96], ## brats_2class max_size : 148,188,156
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')
parser.add_argument('--total_labeled_num', type=int, default=250,
                    help='total labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str,  default='6', help='GPU to use')
parser.add_argument('--T', type=int,  default=8, help='ATD iter')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="SSL_Brats19_2class", config={}, reinit=True)
wandb.run.name = '{}/{}_{}'.format(args.exp,args.model,args.max_iterations)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


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

    def class_create(input_tensor,n_classes):
        tensor_list = []
        for i in range(n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()


    db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.total_labeled_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001) #weight_decay=0.0005
    ce_loss = CrossEntropyLoss()        ##torch
    dice_loss = losses.DiceLoss(args.num_classes)       ##make
    mse_loss = MSELoss()        ##torch

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]     ##unlabel

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
                                                                2 * stride * i:2 * stride * (i + 1)] = ema_model(
                        ema_inputs)  ## unlabel의 shape를 *2로 복사한 값에 새로운 noise를 >계속 넣어줘서 teacher model에 계속 실험  -> preds에 계속 결과 쌓기
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, d, w, h)  # 8,bs,ch,w,h,d
            seg_avg = torch.mean(preds, dim=0)  # 2,2,w,h,d

            SDM_preds = SDM_preds.reshape(T, stride, 1, d, w, h)
            SDM_avg = torch.mean(SDM_preds, dim=0)

            ##entropy 가 낮으면 확실, uncertainty 값이 낮아지길 원함

            loss_ce = ce_loss(outputs_2class[:labeled_bs], label_batch[:labeled_bs])
            loss_dice = dice_loss(outputs_soft_2class[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1).float())
            loss_sdf = mse_loss(outputs_1class_tanh[:labeled_bs, 0, ...], gt_dis)

            supervised_loss = 0.5 * (loss_dice + loss_ce ) + loss_sdf      ## (추측)supervised : SDM의 영향 > segmentation 영향 -> SDM loss 적게 반영

            #####################################################################################################unsupervised loss
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            consistency_mse = torch.mean((outputs_soft_2class[labeled_bs:] - seg_avg) ** 2)  # (batch, 2, 112,112,80)

            consistency_1class_loss = mse_loss(outputs_1class_tanh[labeled_bs:], SDM_avg)  ## (2,1,96,96,96), SDM loss가 80배정도 큼

            #consistency_dice = (
            #    dice_loss(outputs_soft_2class[labeled_bs:], seg_avg, target_one_hot_encoder=False))


            loss = supervised_loss + consistency_weight * (consistency_1class_loss + consistency_mse)

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

            #logging.info(
            #    'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, consistency_loss:%f, threshold_up_rate:%f' %
            #    (iter_num, loss.item(), loss_ce.item(), loss_dice.item(),consistency_loss,\
            #     (ramps.sigmoid_rampup(iter_num,max_iterations))))

            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "loss_ce": loss_ce.item(),
                "loss_dice": loss_dice.item(),
                "loss_sdf_1class" : loss_sdf.item(),
                "consistency_segMse_loss" : consistency_mse.item(),
                "consistency_1class_loss":consistency_1class_loss.item(),
                "consistency_weight":consistency_weight

            })



            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=32, stride_z=32)
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

    snapshot_path = '/data/sohui/BraTS/data/brats_2class_train_result/BraTs19_label_{}_{}/{}/{}_{}_T={}'\
        .format(args.labeled_num, args.total_labeled_num,args.exp,args.model,args.max_iterations,args.T)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('../code', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)