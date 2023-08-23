import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from itertools import cycle

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_sdf import VNet
from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.ATLAS.ATLAS2022_label import (ATLAS2022, RandomCrop, CenterCrop,
                                               RandomRotFlip, ToTensor,
                                               TwoStreamBatchSampler)
from dataloaders.ATLAS.ATLAS2022_unlabel import (ATLAS2022_unlabel, RandomCrop_unlabel, CenterCrop_unlabel,
                                               RandomRotFlip_unlabel, ToTensor_unlabel,
                                               TwoStreamBatchSampler)
from utils.util import compute_sdf
from ATLAS_val_3D_2task import test_all_case


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/ATLAS/data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='test', help='model_name')
parser.add_argument('--model', type=str,
                    default='test')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float, default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--labeled_num', type=int, default=68, help='random seed')
parser.add_argument('--total_labeled_num', type=int, default=612,
                    help='total labeled data')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--patch_size', type=list,  default=[112,112,112], ## brats_2class max_size : 148,188,156
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--consistency_weight', type=float, default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str, default='5', help='GPU to use')
parser.add_argument('--beta', type=float, default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path

snapshot_path = '/data/sohui/ATLAS/ATLAS_2class_train_result/ATLAS_label_{}_{}/{}/{}_{}'.\
    format(args.labeled_num,args.total_labeled_num,args.exp,args.model,args.max_iterations)


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False  # True #
    cudnn.deterministic = True  # False #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes =  args.num_classes
patch_size =  args.patch_size


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    # make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes - 1,
                   normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model()


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train_label = ATLAS2022(base_dir=train_data_path,  # 62 data
                               split='train',
                               num=None,
                               transform=transforms.Compose([
                                   RandomRotFlip(),
                                   RandomCrop(args.patch_size),
                                   ToTensor(),

                               ]))

    db_train_unlabel = ATLAS2022_unlabel(base_dir=train_data_path,  # 550 data
                                         split='train',
                                         num=None,
                                         transform=transforms.Compose([
                                             RandomRotFlip_unlabel(),
                                             RandomCrop_unlabel(args.patch_size),
                                             ToTensor_unlabel(),

                                         ]))

    trainloader_label = DataLoader(db_train_label, batch_size=args.labeled_bs, shuffle=False,
                                   num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    trainloader_unlabel = DataLoader(db_train_unlabel, batch_size=args.batch_size - args.labeled_bs, shuffle=False,
                                     num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    labeled_data, unlabeled_data = (cycle(trainloader_label), trainloader_unlabel)

    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = BCEWithLogitsLoss()
    mse_loss = MSELoss()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(unlabeled_data)))

    iter_num = 0
    max_epoch = max_iterations // len(unlabeled_data) + 1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(zip(labeled_data,unlabeled_data)):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch[0]['image'], sampled_batch[0]['label']
            volume_batch = torch.cat((volume_batch, sampled_batch[1]['image']), dim=0)

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs_tanh, outputs = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)

            # calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:].cpu(
                ).numpy(), outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            loss_sdf = mse_loss(outputs_tanh[:labeled_bs, 0, ...], gt_dis)
            loss_seg = ce_loss(
                outputs[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_seg_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            dis_to_mask = torch.sigmoid(-1500 * outputs_tanh)

            consistency_loss = torch.mean((dis_to_mask - outputs_soft) ** 2)
            supervised_loss = loss_seg_dice + args.beta * loss_sdf
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(torch.argmax(
                outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_consis: %f, loss_haus: %f, loss_seg: %f, loss_dice: %f' %
                (iter_num, loss.item(), consistency_loss.item(), loss_sdf.item(),
                 loss_seg.item(), loss_seg_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('iteration %d : loss : %f' % (iter_num, loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = dis_to_mask[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Dis2Mask', grid_image, iter_num)

                image = outputs_tanh[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/DistMap', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

                image = gt_dis[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_DistMap',
                                 grid_image, iter_num)

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num > 0 and iter_num % 500 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="val.txt", num_classes=args.num_classes, patch_size=args.patch_size,
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
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()