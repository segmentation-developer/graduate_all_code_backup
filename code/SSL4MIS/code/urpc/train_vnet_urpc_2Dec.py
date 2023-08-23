import argparse
import logging
import os
import random
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from code.dataloaders.LA.la_heart_modi import LAHeart, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.vnet_concat_dv_semi_2Dec import VNet
#from networks.unet_3D_dv_semi import unet_3D_dv_semi
from utils import losses, ramps
from code.urpc.validation.val_urpc_util_2Dec import test_all_case
import wandb
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/LA_dataset/2018LA_Seg_TrainingSet', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='LA/test', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=50000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.1,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size',nargs='+', type=int,  default=[112,112,80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=2,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=18,
                    help='labeled data')
parser.add_argument('--total_labeled_num', type=int, default=80,
                    help='total labeled data')
parser.add_argument('--total_num', type=int, default=100,
                    help='total data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=400.0, help='consistency_rampup')
parser.add_argument('--gpu', type=str,  default='2,5,6,7', help='GPU to use')
parser.add_argument('--add', type=float,  default=1e-8)
parser.add_argument('--fold', type=str,  default='fold1')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def train(args, snapshot_path, fold, train_idx, val_idx):

    num_classes = 2
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = VNet(n_channels= 1, n_classes=num_classes, n_filters=16, normalization='batchnorm', has_dropout=True).cuda()
    model = nn.DataParallel(model).to(device)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    db_train = LAHeart(base_dir=train_data_path,
                       split='total',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(args.patch_size),
                           ToTensor(),
                       ]),
                       train_idx=train_idx)

    labeled_idxs = list(range(train_idx[0], train_idx[args.labeled_num]))
    unlabeled_idxs = list(range(train_idx[args.labeled_num], train_idx[-1] + 1))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)



    model.train()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    kl_distance = nn.KLDivLoss(reduction='none')
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4, outputs_noise1, outputs_noise2, outputs_noise3, outputs_noise4 = model(
                volume_batch)
            #softmax
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
            outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)

            outputs_noise1_soft = torch.softmax(outputs_noise1, dim=1)
            outputs_noise2_soft = torch.softmax(outputs_noise2, dim=1)
            outputs_noise3_soft = torch.softmax(outputs_noise3, dim=1)
            outputs_noise4_soft = torch.softmax(outputs_noise4, dim=1)

            #supervised
            loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux4 = ce_loss(outputs_aux4[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux5 = ce_loss(outputs_noise1[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux6 = ce_loss(outputs_noise2[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux7 = ce_loss(outputs_noise3[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])
            loss_ce_aux8 = ce_loss(outputs_noise4[:args.labeled_bs],
                                   label_batch[:args.labeled_bs])

            loss_dice_aux1 = dice_loss(
                outputs_aux1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux2 = dice_loss(
                outputs_aux2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux3 = dice_loss(
                outputs_aux3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux4 = dice_loss(
                outputs_aux4_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux5 = dice_loss(
                outputs_noise1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux6 = dice_loss(
                outputs_noise2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux7 = dice_loss(
                outputs_noise3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_dice_aux8 = dice_loss(
                outputs_noise4_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

            supervised_loss = (loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3 + loss_ce_aux4 + loss_ce_aux5 + loss_ce_aux6 + loss_ce_aux7 + loss_ce_aux8 +
                               loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3 + loss_dice_aux4 + loss_dice_aux5 + loss_dice_aux6 + loss_dice_aux7 + loss_dice_aux8) / 16

            # 4가지 prediction 합친 평균 값 / log는 왜 붙이는거야?
            '''
            preds = (outputs_aux1_soft +
                     outputs_aux2_soft + outputs_aux3_soft + outputs_aux4_soft) / 4
            '''

            # entropyLoss = losses.entropy_loss(preds)
            # entropyLoss = losses.entropy_minmization(preds)

            variance_aux1 = torch.sum(kl_distance(
                torch.log(outputs_aux1_soft[args.labeled_bs:]), outputs_noise1_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(
                torch.log(outputs_aux2_soft[args.labeled_bs:]), outputs_noise2_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(
                torch.log(outputs_aux3_soft[args.labeled_bs:]), outputs_noise3_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            variance_aux4 = torch.sum(kl_distance(
                torch.log(outputs_aux4_soft[args.labeled_bs:]), outputs_noise4_soft[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            consistency_dist_aux1 = (
                outputs_noise1_soft[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux1 = torch.mean(
                consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + args.add) + torch.mean(
                variance_aux1)

            consistency_dist_aux2 = (
               outputs_noise2_soft[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(
                consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + args.add) + torch.mean(
                variance_aux2)

            consistency_dist_aux3 = (
                outputs_noise3_soft[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(
                consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + args.add) + torch.mean(
                variance_aux3)

            consistency_dist_aux4 = (
                outputs_noise4_soft[args.labeled_bs:] - outputs_aux4_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux4 = torch.mean(
                consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + args.add) + torch.mean(
                variance_aux4)

            consistency_loss = (consistency_loss_aux1 +
                                consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4
            loss = supervised_loss + consistency_weight * consistency_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/supervised_loss',
                              supervised_loss, iter_num)
            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "seg_main_loss": supervised_loss.item(),
                "dec1_consistency_loss": consistency_loss_aux1.item(),
                "dec2_consistency_loss": consistency_loss_aux2.item(),
                "dec3_consistency_loss": consistency_loss_aux3.item(),
                "dec4_consistency_loss": consistency_loss_aux4.item(),
                # "random_drop_loss": random_drop_loss.item(),
                # "feature_drop_loss": feature_drop_loss.item(),
                # "noise_loss": noise_loss.item(),
                # "sl_auxDec_concat_loss" : sl_auxDec_loss.item(),
                # "unsl_auxDec_concat_loss": unsl_auxDec_loss.item(),
                # "dice_metrics": dc.item()

            })

            logging.info(
                '[fold : %d] iteration %d : loss : %f, supervised_loss: %f, consistency_loss_aux1: %f, consistency_loss_aux2: %f, consistency_loss_aux3: %f, consistency_loss_aux4: %f,\
                    consistency_dist_aux1 : %f, exp_variance_aux1: %f, variance_aux1: %f, variance_aux2: %f, variance_aux3: %f,variance_aux4: %f' %
                (fold, iter_num, loss.item(), supervised_loss.item(), consistency_loss_aux1.item(),
                 consistency_loss_aux2.item(), consistency_loss_aux3.item(), consistency_loss_aux4.item(), \
                 torch.mean(consistency_dist_aux1).item(), torch.mean(exp_variance_aux1).item(),
                 torch.mean(variance_aux1).item(), torch.mean(variance_aux2).item(),
                 torch.mean(variance_aux3).item(),
                 torch.mean(variance_aux4).item()))

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = torch.argmax(outputs_aux1_soft, dim=1, keepdim=True)[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 1000 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model, args.root_path, test_list="total.list", num_classes=num_classes,
                    patch_size=args.patch_size,
                    stride_xy=18, stride_z=4, val_idx=val_idx)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'fold_{}_iter_{}_dice_{}.pth'.format(
                                                      fold, iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             'fold_{}_{}_best_model.pth'.format(fold, args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                for cls in range(1, num_classes):
                    writer.add_scalar('info/val_cls_{}_dice_score'.format(cls),
                                      avg_metric[cls - 1, 0], iter_num)
                    writer.add_scalar('info/val_cls_{}_hd95'.format(cls),
                                      avg_metric[cls - 1, 1], iter_num)
                writer.add_scalar('info/val_mean_dice_score',
                                  avg_metric[:, 0].mean(), iter_num)
                writer.add_scalar('info/val_mean_hd95',
                                  avg_metric[:, 1].mean(), iter_num)

                wandb.log({
                    "dice_score": avg_metric[:, 0].mean(),  # total loss
                    "hd95": avg_metric[:, 1].mean(),

                })

                logging.info(
                    'fold : %d iteration %d : dice_score : %f hd95 : %f' % (
                        fold, iter_num, avg_metric[:, 0].mean(), avg_metric[:, 1].mean()))
                model.train()

            '''
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(snapshot_path,
                                            'fold_{}_iter_{}.pth'.format(
                                                      fold,iter_num))
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
            '''
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

    snapshot_path = "/data/sohui/LA_dataset/{}/{}".format(args.exp, args.model)
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

    kf = KFold(n_splits=5)
    total_path = args.root_path + '/total.list'
    with open(total_path, 'r') as f:
        image_list = f.readlines()

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_list)):
        '''
        if args.fold == 'front':
            if fold < 3:
                wandb.init(project="SSL_LA heart_kfold", config={}, reinit=True)
                wandb.run.name = '{}_fold{}'.format(args.exp, fold)
                print(fold, 'fold')
                print(train_idx)
                print(val_idx)
                train(args, snapshot_path, fold, train_idx, val_idx)


        if args.fold == 'behind':
            if fold == 3 :
                wandb.init(project="SSL_LA heart_kfold", config={}, reinit=True)
                wandb.run.name = '{}_fold{}'.format(args.exp, fold)
                print(fold, 'fold')
                print(train_idx)
                print(val_idx)
                train(args, snapshot_path, fold, train_idx, val_idx)
            if fold > 3:
                wandb.init(project="SSL_LA heart_kfold", config={}, reinit=True)
                wandb.run.name = '{}_fold{}'.format(args.exp, fold)
                print(fold, 'fold')
                print(train_idx)
                print(val_idx)
                train(args, snapshot_path, fold, train_idx, val_idx)
        '''




        if args.fold == 'fold1':
            if fold == 1:
                wandb.init(project="SSL_LA heart_kfold", config={}, reinit=True)
                wandb.run.name = '{}_fold{}'.format(args.exp, fold)
                print(fold, 'fold')
                print(train_idx)
                print(val_idx)
                train(args, snapshot_path, fold, train_idx, val_idx)

        if args.fold == 'fold2':
            if fold == 2:
                wandb.init(project="SSL_LA heart_kfold", config={}, reinit=True)
                wandb.run.name = '{}_fold{}'.format(args.exp, fold)
                print(fold, 'fold')
                print(train_idx)
                print(val_idx)
                train(args, snapshot_path, fold, train_idx, val_idx)

        if args.fold == 'fold0':
            if fold == 0:
                wandb.init(project="SSL_LA heart_kfold", config={}, reinit=True)
                wandb.run.name = '{}_fold{}'.format(args.exp, fold)
                print(fold, 'fold')
                print(train_idx)
                print(val_idx)
                train(args, snapshot_path, fold, train_idx, val_idx)


