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
from dataloaders.brats.brats2019 import (BraTS2019, RandomCrop,
                                         RandomRotFlip, ToTensor,
                                         TwoStreamBatchSampler)
from networks.unet_3D import unet_3D
from networks.vnet import VNet
#from monai.networks.nets import UNETR
#from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, ramps
from brats_val_3D import test_all_case
from skimage import segmentation as skimage_seg
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/BraTS/data/BraTs2019', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='/test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_vnet_3D', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96,96,96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
#parser.add_argument(
#    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )

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
                    default=200.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
parser.add_argument('--add', type=float,  default=1e-8)
parser.add_argument('--fold', type=str,  default=None)
args = parser.parse_args()
#config = get_config(args)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

wandb.init(project="SSL_brats2019", config={}, reinit=True)
wandb.run.name = '{}'.format(args.exp)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
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
        model = unet_3D( in_channels=1, n_classes=num_classes ).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = VNet(n_channels= 1, n_classes=num_classes, n_filters=16, normalization='batchnorm', has_dropout=True).cuda()



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
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    mse_loss = MSELoss()
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            print(label_batch.max())
            with torch.no_grad():
                boundary_label = compute_bound(label_batch.detach().cpu().numpy())
                boundary_label = torch.from_numpy(boundary_label).float().cuda()

            outputs1= model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            boundary1 = compute_bound(outputs_soft1[:,1,:,:,:].detach().cpu().numpy())        #obj channel
            boundary1 = torch.from_numpy(boundary1).float().cuda()

            outputs2= model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            boundary2 = compute_bound(outputs_soft2[:,1,:,:,:].detach().cpu().numpy())
            boundary2 = torch.from_numpy(boundary2).float().cuda()

            '''
            if iter_num == 100:
                plt.figure(figsize=(18, 18))
                # for idx in range(3):
                plt.subplot(6, 1, 1)
                plt.imshow(boundary_label[0][:, :, 55:56].detach().cpu().numpy(), cmap='gray')
                plt.subplot(6, 1, 2)
                plt.imshow(outputs_soft1[0,1,:,:, 55:56].detach().cpu().numpy(), cmap='gray')
                plt.subplot(6, 1, 3)
                plt.imshow(outputs_soft2[0,1,:,:, 55:56].detach().cpu().numpy(), cmap='gray')
                
                plt.tight_layout()
                plt.show()
            '''

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            ##supervised :dice CE
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            boundary_loss1 = torch.mean((boundary1[:args.labeled_bs]-boundary_label[:args.labeled_bs])**2)
            boundary_loss2 = torch.mean((boundary2[:args.labeled_bs]-boundary_label[:args.labeled_bs])**2)
            boundary_loss = boundary_loss1 + boundary_loss2

            ##pseudo label(2,112,112,80)
            pseudo_outputs1 = torch.argmax(
                outputs_soft1[args.labeled_bs:].detach(), dim=1, keepdim=True)
            pseudo_outputs1 = class_create(pseudo_outputs1,num_classes)

            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=True)
            pseudo_outputs2 = class_create(pseudo_outputs2, num_classes)

            '''
            pseudo_supervision1 = 0.5 * (ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2) +
                                         dice_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2.unsqueeze(1)))
            pseudo_supervision2 = 0.5 * (ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1) +
                                         dice_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1.unsqueeze(1)))
            
            variance1 = torch.sum(kl_distance(
                torch.log(outputs_soft1[args.labeled_bs:]), pseudo_outputs2), dim=1, keepdim=True)
            exp_variance1 = torch.exp(-variance1)

            variance2 = torch.sum(kl_distance(
                torch.log(outputs_soft2[args.labeled_bs:]), pseudo_outputs1), dim=1, keepdim=True)
            exp_variance2 = torch.exp(-variance2)

            consistency_dist1 = (
                outputs_soft1[args.labeled_bs:] - pseudo_outputs2) ** 2
            consistency_loss1 = torch.mean(
                consistency_dist1 * exp_variance1) / (torch.mean(exp_variance1) + args.add) + torch.mean(variance1)

            consistency_dist2 = (
                outputs_soft2[args.labeled_bs:] - pseudo_outputs1) ** 2
            consistency_loss2 = torch.mean(
                consistency_dist2 * exp_variance2) / (torch.mean(exp_variance2) + args.add) + torch.mean(variance2)
            '''

            ##unsupervised
            pseudo_supervision1 = ce_loss(outputs1[args.labeled_bs:], pseudo_outputs2)
            pseudo_supervision2 = ce_loss(outputs2[args.labeled_bs:], pseudo_outputs1)

            #mseLoss1 = 1 * mse_loss(outputs_soft1[args.labeled_bs:], pseudo_outputs2)
            #mseLoss2 = 1 * mse_loss(outputs_soft2[args.labeled_bs:], pseudo_outputs1)

            #KL_loss = 0.5 * torch.mean(torch.sum(kl_distance(torch.log(aux_outputs_soft1[args.labeled_bs:]), aux_outputs_soft2[args.labeled_bs:]), dim=1, keepdim=True))



            supervised_loss= loss1 + loss2 + boundary_loss
            unsupervised_loss =  consistency_weight *(pseudo_supervision1 +pseudo_supervision2)


            loss = supervised_loss + unsupervised_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/supervised_loss',
                              supervised_loss, iter_num)
            writer.add_scalar('loss/unsupervised_loss',
                              unsupervised_loss, iter_num)

            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "supervised_loss": supervised_loss.item(),
                "unsupervised_loss": unsupervised_loss.item(),
                "boundary_loss" :  boundary_loss.item(),
                # "random_drop_loss": random_drop_loss.item(),
                # "feature_drop_loss": feature_drop_loss.item(),
                # "noise_loss": noise_loss.item(),
                # "sl_auxDec_concat_loss" : sl_auxDec_loss.item(),
                # "unsl_auxDec_concat_loss": unsl_auxDec_loss.item(),
                # "dice_metrics": dc.item()

            })

            logging.info('iteration %d : supervised_loss : %f boundary_loss : %f unsupervised_loss : %f'  % (
                iter_num, supervised_loss.item(), boundary_loss.item(), unsupervised_loss.item()))


            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft2[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 1000 == 0:
                model1.eval()
                avg_metric1 = test_all_case(
                    model1, args.root_path, test_list="val.txt", num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric1[:, 0].mean() > best_performance1:
                    best_performance1 = avg_metric1[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                writer.add_scalar('info/model1_val_dice_score',
                                  avg_metric1[0, 0], iter_num)
                writer.add_scalar('info/model1_val_hd95',
                                  avg_metric1[0, 1], iter_num)
                logging.info(
                    'iteration %d : model1_dice_score : %f model1_hd95 : %f' % (
                        iter_num, avg_metric1[0, 0].mean(), avg_metric1[0, 1].mean()))
                model1.train()

                model2.eval()
                avg_metric2 = test_all_case(
                    model2, args.root_path, test_list="val.txt", num_classes=args.num_classes, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                if avg_metric2[:, 0].mean() > best_performance2:
                    best_performance2 = avg_metric2[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                writer.add_scalar('info/model2_val_dice_score',
                                  avg_metric2[0, 0], iter_num)
                writer.add_scalar('info/model2_val_hd95',
                                  avg_metric2[0, 1], iter_num)
                logging.info(
                    'iteration %d : model2_dice_score : %f model2_hd95 : %f' % (
                        iter_num, avg_metric2[0, 0].mean(), avg_metric2[0, 1].mean()))
                model2.train()

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                for param_group in optimizer2.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 10000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


    snapshot_path = "/data/sohui/BraTS/train_result/BraTs19_label_{}_{}/{}".format(args.labeled_num, args.total_labeled_num,args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
