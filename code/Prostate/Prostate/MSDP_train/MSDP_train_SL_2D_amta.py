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
#   download pre-trained denseUnet_3D to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

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
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# from config import get_config
from Prostate.dataloaders.preprocess.MSD_prostate.dataset_2D import BaseDataSets, RandomGenerator
from Prostate.networks.amtaNet_2D import AMTA_Net
from Prostate.utils import ramps, losses
from MSDP_val_2D import test_single_volume
from skimage import segmentation as skimage_seg

from monai.inferers import sliding_window_inference
# from monai.networks.nets import UNet

from monai.data import (
    DataLoader,
    )
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/MSD_prostate/Task05_Prostate_pre_slices', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='prostate_result_2D/test', help='experiment_name')
parser.add_argument('--denseUnet_3D', type=str,
                    default='unet_20000_1', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[240, 240],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=3,
                    help='output channel of network')
# parser.add_argument(
#    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=24,
                    help='labeled data')
parser.add_argument('--total_labeled_num', type=int, default=242,
                    help='total labeled data')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str, default='4,5,6,7', help='GPU to use')
parser.add_argument('--add', type=float, default=1e-8)
parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--sw_batch_size', type=int, default=16)
parser.add_argument('--overlap', type=float, default=0.5)
args = parser.parse_args()
# config = get_config(args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.init(project="SSL_MSD_Prostate", config={}, reinit=True)
wandb.run.name = '{}/{}'.format(args.exp, args.model)

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
def update_ema_variables(denseUnet_3D, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in (ema_model.parameters(), denseUnet_3D.parameters()):
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
        # denseUnet_3D = Attention_UNet(n_classes=num_classes, in_channels= 1)
        # denseUnet_3D = unet_3D(n_classes=num_classes, in_channels=1)       #기존 unet
        model = AMTA_Net(in_ch=1)
        # denseUnet_3D = VNet(n_channels=1, n_classes=num_classes, n_filters=16, normalization='batchnorm',has_dropout=True)    # 기존 vnet
        '''
        denseUnet_3D = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        ).cuda()
        '''

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model1 = nn.DataParallel(model1).to(device)

    model1.train()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    def class_create(input_tensor, n_classes):
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

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator([320, 320])]),
        output_size=[320,320])

    db_val = BaseDataSets(base_dir=args.root_path, split="val",output_size=[320,320])

    trainloader = DataLoader(db_train, batch_size=args.labeled_bs, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)
    # dice_loss_tz = losses.DiceLoss(args.num_classes-1)
    dice_loss_class = losses.DiceLoss_class(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    # logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    lr_ = base_lr

    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):  # 0,1 : SL_trainloader(bs:1), UL_trainloader(bs:1)
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            # label_batch = label_batch.double()
            # label_batch_tz = torch.where(label_batch == 1., 0., label_batch)

            roi_size = args.patch_size
            sw_batch_size = args.sw_batch_size
            '''
            plt.figure(figsize=(18, 18))
            # for idx in range(3):
            plt.subplot(8, 1, 1)
            plt.imshow(volume_batch[0][0][:, :, 20:21].detach().cpu())
            plt.subplot(8, 1, 2)
            plt.imshow(volume_batch[1][0][:, :, 20:21].detach().cpu())
            plt.subplot(8, 1, 3)
            plt.imshow(volume_batch[2][0][:, :, 20:21].detach().cpu())
            plt.subplot(8, 1, 4)
            plt.imshow(volume_batch[3][0][:, :, 20:21].detach().cpu())
            plt.subplot(8, 1, 5)
            plt.imshow(volume_batch[4][0][:, :, 20:21].detach().cpu())
            plt.subplot(8, 1, 6)
            plt.imshow(volume_batch[5][0][:, :, 20:21].detach().cpu())
            plt.subplot(8, 1, 7)
            plt.imshow(volume_batch[6][0][:, :, 20:21].detach().cpu())
            plt.subplot(8, 1, 8)
            plt.imshow(volume_batch[7][0][:, :, 30:31].detach().cpu())
            plt.tight_layout()
            plt.show()
            print()
            '''
            # noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2,0.2)
            # volume_batch = volume_batch + noise

            att_out, first_out = sliding_window_inference(
                volume_batch, roi_size, sw_batch_size, model1, overlap=args.overlap)
            att_out_soft = torch.softmax(att_out, dim=1)
            first_out_soft = torch.softmax(first_out, dim=1)

            # label_batch = label_batch.double()
            # label_batch = torch.where(label_batch == 2., 0., label_batch)

            # with torch.no_grad():
            #    boundary_label = compute_bound(label_batch.detach().cpu().numpy())
            #    boundary_label = torch.from_numpy(boundary_label).float().cuda()

            # boundary1 = compute_bound(dsv1_F_soft[:, 1, :, :, :].detach().cpu().numpy())  # obj channel
            # boundary1 = torch.from_numpy(boundary1).float().cuda()

            # boundary2 = compute_bound(dsv1_S_soft[:, 1, :, :, :].detach().cpu().numpy())
            # boundary2 = torch.from_numpy(boundary2).float().cuda()

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            ##supervised :dice CE
            loss1 = (ce_loss(att_out[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                att_out_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = (ce_loss(first_out[:args.labeled_bs], label_batch[:args.labeled_bs].long()) + dice_loss(
                first_out_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            # imgAddBound1= boundary1[:args.labeled_bs]+dsv1_F_soft[:args.labeled_bs, 1, :, :, :]
            # imgAddBound2 = boundary2[:args.labeled_bs] +dsv1_S_soft[:args.labeled_bs, 1, :, :, :]
            # gtAddBound = boundary_label[:args.labeled_bs] + label_batch[:args.labeled_bs]
            # boundary_loss1 = torch.mean((imgAddBound1 - gtAddBound) ** 2)
            # boundary_loss2 = torch.mean((imgAddBound2 - gtAddBound) ** 2)
            # boundary_loss = boundary_loss1 + boundary_loss2

            ##unsupervised
            '''
            variance_aux1 = torch.sum(kl_distance(
                torch.log(dsv1_F_soft[args.labeled_bs:]), dsv1_S_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)


            variance_aux2 = torch.sum(kl_distance(
                torch.log(dsv2_F_soft[args.labeled_bs:]), dsv2_S_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(
                torch.log(dsv3_F_soft[args.labeled_bs:]), dsv3_S_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            variance_aux4 = torch.sum(kl_distance(
                torch.log(dsv4_F_soft[args.labeled_bs:]), dsv4_S_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)
            '''

            ########consis loss
            '''
            consistency_dist_aux1 = (
               dsv1_F_soft[args.labeled_bs:] - dsv1_S_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux1 = torch.mean(
                consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(
                variance_aux1)


            consistency_dist_aux2 = (
                dsv2_F_soft[args.labeled_bs:] - dsv2_S_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(
                consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(
                variance_aux2)

            consistency_dist_aux3 = (
                dsv3_F_soft[args.labeled_bs:] - dsv3_S_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(
                consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(
                variance_aux3)

            consistency_dist_aux4 = (
                dsv4_F_soft[args.labeled_bs:] - dsv4_S_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux4 = torch.mean(
                consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(
                variance_aux4)
                '''

            # enc5_contrastive_loss = info_nce_loss(features1 = fc5_F_soft[args.labeled_bs:], features2 = fc5_S_soft[args.labeled_bs:])

            # dsv1_F_norm = F.normalize(head1_F.permute(1, 0) , dim=1) # bs,ch -> ch ,bs     #batch size에 대해  normalize
            # dsv1_S_norm = F.normalize(head1_S.permute(1, 0) , dim=1)
            # head1_contrastive_loss = nt_xent_criterion_ch(dsv1_F_norm[:,args.labeled_bs:], dsv1_S_norm[:,args.labeled_bs:] )

            supervised_loss = (loss1 + loss2)

            loss = supervised_loss

            optimizer1.zero_grad()
            loss.backward()

            optimizer1.step()

            iter_num = iter_num + 1

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/supervised_loss',
                              supervised_loss, iter_num)

            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "att_loss": loss1.item(),
                "first_loss": loss2.item(),

            })

            logging.info('iteration %d : supervised_loss : %f att_loss : %f  first_loss : %f' % (
                iter_num, supervised_loss.item(), loss1.item(), loss2.item()))

            '''
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = dsv1_F_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Model1_Predicted_label',
                                 grid_image, iter_num)

                image = dsv1_S_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Model2_Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)
                '''

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

            if iter_num > 0 and iter_num % 10 == 0:
                model1.eval()
                metric_list = 0.0
                ith = 0
                for i_batch, sampled_batch in enumerate(valloader):
                    if sampled_batch["label"].sum() == 0 :
                        continue
                    else:
                        metric_i = test_single_volume(
                            sampled_batch["image"], sampled_batch["label"], model1, classes=args.num_classes,
                            patch_size=[320, 320])
                        ith += 1
                        metric_list += np.array(metric_i)

                metric_list = metric_list / ith
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model1.train()

            if iter_num % 5000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":

    snapshot_path = "/data/sohui/MSD_prostate/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('../..', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
