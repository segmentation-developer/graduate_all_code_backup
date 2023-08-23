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
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.la_heart import LAHeart, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.vnet_auxDec_concat_DenseM_decPointwise_concat_addconv import VNet
from utils import losses, metrics, ramps
from val_3D import test_all_case
import wandb


wandb.init(project="SSL_LA heart2", config ={
        #"batch size" :16,
        #"max_iterations":5000,
        #"encoder": 'resnet-34',
        #"weights": 'imagenet',
        "mdoel" :"supervised",
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
                    default='LA', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='test', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[112, 112, 80],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')
parser.add_argument('--gpu', type=str,  default='6', help='GPU to use')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

patch_size = args.patch_size


def train(args, check_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    model = VNet().cuda()  # seg class = 2
    db_train = LAHeart(base_dir=train_data_path,            #h5py 때문에 [80,138,200]으로
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(check_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs = model(volume_batch)

            #print(outputs.size())
            #print(label_batch.size())

            outputs_soft = torch.softmax(outputs, dim=1)
            #outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            #outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)

            loss_ce = ce_loss(outputs, label_batch)
            #loss_ce_aux1 = ce_loss(outputs_aux1,label_batch.long())
            #loss_ce_aux2 = ce_loss(outputs_aux2, label_batch.long())

            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            #loss_dice_aux1 = dice_loss(outputs_aux1_soft, label_batch.unsqueeze(1))
           # loss_dice_aux2 = dice_loss(outputs_aux2_soft, label_batch.unsqueeze(1))

            '''
            print(outputs.size())       #[2, 2, 112, 112, 80]
            print(label_batch.size())       #[2, 112, 112, 80]
            print(outputs_soft.size())      #[2, 2, 112, 112, 80]
            #print(label_batch.unsqueeze(1).detach().cpu().numpy())      #[2, 1, 112, 112, 80]
            print(loss_dice)        # scalar 0.4165
            print(loss_ce)          # scalar 0.7351
            '''

            loss = (loss_dice + loss_ce) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)


            wandb.log({
                "iter": iter_num,
                "total_loss": loss.item(),  # total loss
                "seg_main_loss": loss_ce.item()+loss_dice.item(),
                #"seg_noise_loss": loss_ce_aux1.item()+loss_dice_aux1.item(),
                #"seg_featureDrop_loss": loss_ce_aux2.item() + loss_dice_aux2.item()
                # "random_drop_loss": random_drop_loss.item(),
                # "feature_drop_loss": feature_drop_loss.item(),
                # "noise_loss": noise_loss.item(),
                # "sl_auxDec_concat_loss" : sl_auxDec_loss.item(),
                # "unsl_auxDec_concat_loss": unsl_auxDec_loss.item(),
                # "dice_metrics": dc.item()

            })


            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
            writer.add_scalar('loss/loss', loss, iter_num)

            if iter_num % 20 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label',
                                 grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label',
                                 grid_image, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case(
                    model , args.root_path, test_list="test.list", num_classes=2, patch_size=args.patch_size,
                    stride_xy=18, stride_z=4)           #h5py = (179,137,88)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(check_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(check_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)

                wandb.log({
                    "iter": iter_num,
                    "val_dice":avg_metric[0, 0],
                    "val-95hd":avg_metric[0, 1]
                    # "feature_drop_loss": feature_drop_loss.item(),
                    # "noise_loss": noise_loss.item(),
                    # "sl_auxDec_concat_loss" : sl_auxDec_loss.item(),
                    # "unsl_auxDec_concat_loss": unsl_auxDec_loss.item(),
                    # "dice_metrics": dc.item()

                })

                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f' % (iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    check_path, 'iter_' + str(iter_num) + '.pth')
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

    check_path = "/data/sohui/LA_dataset/{}/{}".format(args.exp, args.model)
    if not os.path.exists(check_path):
        os.makedirs(check_path)
    if os.path.exists(check_path + '/code'):
        shutil.rmtree(check_path + '/code')
    shutil.copytree('..', check_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=check_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, check_path)
