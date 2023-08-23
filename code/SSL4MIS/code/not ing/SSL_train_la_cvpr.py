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

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet_CCT import VNet
#from networks.discriminator import FC3DDiscriminator

from dataloaders import utils
from utils import ramps, losses, metrics
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from utils.util import compute_sdf
from val_util import test_all_case

import wandb
wandb.init(project="SSL_LA heart3", config ={
        #"batch size" :16,
        #"max_iterations":5000,
        #"encoder": 'resnet-34',
        #"weights": 'imagenet',
        "mdoel" :"DTC",
        #"aspp":"2,3,4,5",
        #"decoder_use_batchnorm" :" yes",
       #"decoder_attention" : " X ",
        #"padding" : "constant",
        "server" : 57,
        "image size": "112,112,80"
    })

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/LA_dataset/2018LA_Seg_TrainingSet', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA', help='model_name')
parser.add_argument('--model', type=str,
                    default='SSL_2perturb_cvpr_URPC', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='maximum epoch number to train')
parser.add_argument('--D_lr', type=float,  default=1e-4,
                    help='maximum discriminator learning rate to train')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='random seed')
parser.add_argument('--seed', type=int,  default=1330, help='random seed')
parser.add_argument('--consistency_weight', type=float,  default=0.1,
                    help='balance factor to control supervised loss and consistency loss')
parser.add_argument('--gpu', type=str,  default='4', help='GPU to use')
parser.add_argument('--beta', type=float,  default=0.3,
                    help='balance factor to control regional and sdm loss')
parser.add_argument('--gamma', type=float,  default=0.5,
                    help='balance factor to control supervised and consistency loss')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="kl", help='consistency_type')
parser.add_argument('--with_cons', type=str,
                    default="without_cons", help='with or without consistency')
parser.add_argument('--consistency', type=float,
                    default=1.0, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=40.0, help='consistency_rampup')
parser.add_argument('--detail', type=int,  default=1,
                    help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0,
                    help='apply NMS post-procssing?')
args = parser.parse_args()

train_data_path = args.root_path

snapshot_path ="/data/sohui/LA_dataset/{}/{}".format(args.exp, args.model)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

num_classes = 1
patch_size = (112, 112, 80)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


if __name__ == "__main__":
    # make logger file
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

    def create_model(ema=False):
        # Network definition
        model = VNet().cuda()
        model = nn.DataParallel(model)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()

    db_train = LAHeart(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    labelnum = args.labelnum    # default 16
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, 80))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

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

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, outputs_aux1, outputs_aux2 = model(volume_batch)
            outputs_soft = torch.sigmoid(outputs)
            outputs_aux1_soft = torch.sigmoid(outputs_aux1)
            outputs_aux2_soft = torch.sigmoid(outputs_aux2)
            #outputs_aux3_soft = torch.sigmoid(outputs_aux3)

            '''
            # calculate the loss
            with torch.no_grad():
                gt_dis = compute_sdf(label_batch[:].cpu(
                ).numpy(), outputs[:labeled_bs, 0, ...].shape)
                gt_dis = torch.from_numpy(gt_dis).float().cuda()
            '''

            loss_ce = ce_loss(
                outputs[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_ce_aux1 = ce_loss(
                outputs_aux1[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            loss_ce_aux2 = ce_loss(
                outputs_aux2[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())
            #loss_ce_aux3 = ce_loss(
            #    outputs_aux3[:labeled_bs, 0, ...], label_batch[:labeled_bs].float())


            loss_dice = losses.dice_loss(
                outputs_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            loss_dice_aux1 = losses.dice_loss(
                outputs_aux1_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            loss_dice_aux2 = losses.dice_loss(
                outputs_aux2_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)
            #loss_dice_aux3 = losses.dice_loss(
            #    outputs_aux3_soft[:labeled_bs, 0, :, :, :], label_batch[:labeled_bs] == 1)

            #dis_to_mask = torch.sigmoid(-1500*outputs_tanh)

            main_supervised_loss = (loss_ce + loss_dice)
            supervised_loss = (loss_ce + loss_dice + loss_ce_aux1 + loss_ce_aux2 +loss_dice_aux1 + loss_dice_aux2 )*(1/6)

            consistency_loss_aux1 = torch.mean(
                (outputs_soft[labeled_bs:, ...] - outputs_aux1_soft[labeled_bs:, ...]) ** 2)
            consistency_loss_aux2 = torch.mean(
                (outputs_soft[labeled_bs:, ...] - outputs_aux2_soft[labeled_bs:, ...]) ** 2)
            #consistency_loss_aux3 = torch.mean(
            #    (outputs_soft[labeled_bs:, ...] - outputs_aux3_soft[labeled_bs:, ...]) ** 2)

            consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2) / 2

            consistency_weight = get_current_consistency_weight(iter_num//150)

            loss = supervised_loss + consistency_weight * consistency_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            dc = metrics.dice(torch.argmax(
                outputs_soft[:labeled_bs], dim=1), label_batch[:labeled_bs])

            iter_num = iter_num + 1
            '''
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss_hausdorff', loss_sdf, iter_num)
            writer.add_scalar('loss/consistency_weight',
                              consistency_weight, iter_num)
            writer.add_scalar('loss/consistency_loss',
                              consistency_loss, iter_num)
            '''

            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "SL_main_loss": loss_dice.item() + loss_ce.item(),  # segmentation loss :dice  :super
                "SL_aux1(noise)_loss": loss_dice_aux1.item() + loss_ce_aux1.item(),
                "SL_aux2(FD)_loss": loss_dice_aux2.item() + loss_ce_aux2.item(),
                #"SL_aux3(RD)_loss": loss_dice_aux3.item() + loss_ce_aux3.item(),
                "SSL_aux1_loss": consistency_loss_aux1.item(),
                "SSL_aux2_loss": consistency_loss_aux2.item(),
                #"SSL_aux3_loss": consistency_loss_aux3.item(),
                "consistency_weight": consistency_weight,
                "consistency_loss": consistency_loss.item(),  # consis loss  : unsuper +super


            })



            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()

                with open(args.root_path + '/test.list', 'r') as f:
                    image_list = f.readlines()
                image_list = [args.root_path + "/" + item.replace('\n', '') + "/mri_norm2.h5" for item in
                              image_list]


                avg_metric = test_all_case(
                    model , image_list, num_classes=num_classes,
                               patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                               save_result=False,
                               metric_detail=args.detail, nms=args.nms)           #h5py = (179,137,88)


                if avg_metric[0] > best_performance:
                    best_performance = avg_metric[0]
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.module.state_dict(), save_mode_path)
                    torch.save(model.module.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0], iter_num)

                wandb.log({
                    "iter": iter_num,
                    "val_dice":avg_metric[0],
                    "va_jacc":avg_metric[1],
                    "val_95": avg_metric[2],
                    "val_ASD": avg_metric[3],
                })


                model.train()


            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.module.state_dict(), save_mode_path)
                #logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
