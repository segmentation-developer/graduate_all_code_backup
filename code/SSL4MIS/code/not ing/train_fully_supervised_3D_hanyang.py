import argparse
import logging
import os
import random
import shutil
import sys
import time
import matplotlib.pyplot as plt

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
#from dataloaders.hanyang_brain.hanyang_brain_h5py import hBrain, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
from networks.vnet import VNet
from utils import losses, metrics, ramps
from val_3D import test_all_case
import wandb


from monai.transforms import (
    AsDiscrete,
    AddChanneld,
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
    ToTensord,
)
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


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
        "image size": "512,512,32"
    })

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/sohui/hanyang_brain/nii_hBrain_unite', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='hanyang_brain', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='vnet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512, 512, 32],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1330, help='random seed')
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "1"


args = parser.parse_args()

patch_size = (512, 512, 32)

directory = '/data/sohui/wonjun_processing/cache_dir'
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)





def train(args, snapshot_path):

    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 3
    model = VNet(n_channels=1, n_classes=num_classes,
                   normalization='batchnorm', has_dropout=True).cuda()
    '''
    db_train = hBrain(base_dir=train_data_path,
                       split='train',  # train/val split
                       transform=transforms.Compose([
                           RandomRotFlip(),
                           RandomCrop(patch_size),
                           ToTensor(),
                       ]))
                       '''
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),

            # Spacingd(
            #    keys=["image", "label"],
            #    pixdim=(1.5, 1.5 , 2.0),             # 2.0: 원본 , 1.5 : brain smaller
            #    mode=("bilinear", "nearest"),
            # ),

            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(256, 256, 32),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=0.10,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=0.10,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.50,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            # Spacingd(
            #    keys=["image", "label"],
            #    pixdim=(1.5, 1.5, 2.0),
            #    mode=("bilinear", "nearest"),
            # ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # RandCropByPosNegLabeld(
            #            keys=["image", "label"],
            #            label_key="label",
            #            spatial_size=(128, 128, 8),
            #            pos=1,
            #            neg=1,
            #            num_samples=4,
            #            image_key="image",
            #            image_threshold=0,
            #        ),
            ToTensord(keys=["image", "label"]),
        ]
    )

    data_dir = '/data/sohui/wonjun_processing'
    split_JSON = "/dataset_70.json"
    datasets = data_dir + split_JSON
    datalist = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "train_validation_test")

    train_ds = CacheDataset(
        data=datalist,
        transform=train_transforms,
        cache_num=24,
        cache_rate=1.0,
        num_workers=8,
    )
    trainloader = DataLoader(
        train_ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=True
    )

    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_num=6,
        cache_rate=1.0,
        num_workers=4
    )

    valloader = DataLoader(
        val_ds, batch_size=2, shuffle=False, num_workers=4, pin_memory=True
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    #trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
    #                         num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1      #MAX_iter =6000, trainloader =154 , max_epoch=39
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=80)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):   #trainloader=70

            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            #if i_batch ==
            plt.figure("mask", (18, 18))
            plt.subplot(2, 2, 1)
            plt.title("bs=1")
            plt.imshow(label_batch[0][:, :, 23:24].detach().cpu())
            '''
            plt.subplot(2, 2, 2)
            plt.title("bs=1")
            plt.imshow(label_batch[2][1][:, :, 23:24].detach().cpu(), cmap='gray')
            plt.subplot(2, 2, 3)
            plt.title("bs=1")
            plt.imshow(label_batch[2][2][:, :, 23:24].detach().cpu(), cmap='gray')'''

            plt.show()



            outputs = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs_soft, label_batch)
            loss = 0.5 * (loss_dice + loss_ce)
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
                "seg_ce_loss": loss_ce.item(),
                "seg_dice_loss": loss_dice.item()
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
                    model, val_loader=valloader, num_classes=3, patch_size=args.patch_size,
                    stride_xy=32, stride_z=4)
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

    snapshot_path = "../model/{}/{}".format(args.exp, args.model)
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
