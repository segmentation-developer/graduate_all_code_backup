import argparse
import os
import shutil
import numpy as np
import torch
from networks.vnet import VNet
from Prostate_test_3D_util import test_all_case
import torch.nn as nn


from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    Spacingd,
    CenterSpatialCropd
)



from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)



def Inference(args,device):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAI"),   #LPS ->
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
            #CenterSpatialCropd(keys=['image', 'label'], roi_size=(176,176,176)),
            #SpatialPadd(keys=["image", "label"], spatial_size=(320, 320, 32), mode="constant"),
        ]
    )

    if args.class_name == 1:
        datasets = args.root_path + "/dataset_fold{}.json".format(args.fold)
        print("total_prostate train : dataset.json")
    if args.class_name == 2:
        datasets = args.root_path + "/dataset_2_fold{}.json".format(args.fold)
        print("transition zone train :dataset_2.json")

    val_files = load_decathlon_datalist(datasets, True, "test")

    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    if args.class_name == 1:
        snapshot_path = "/data/sohui/Prostate/prostate_1c_train_result/{}/{}".format(args.exp, args.model)
        #snapshot_path = "/data/sohui/BraTS/data/brats_2class_train_result/BraTs19_label_40_290/{}/{}".format(args.exp, args.model)
    elif args.class_name == 2:
        snapshot_path = "/data/sohui/Prostate/TZ_1c_train_result/{}/{}".format(args.exp, args.model)
    num_classes = args.num_classes

    if args.class_name == 1:
        test_save_path = "/data/sohui/Prostate/prostate_1c_test_result/{}/{}".format(args.exp, args.model)
    elif args.class_name == 2:
        test_save_path = "/data/sohui/Prostate/TZ_1c_test_result/{}/{}".format(args.exp, args.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    #net = unet_3D(n_classes=num_classes, in_channels=1)
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    if len(args.gpu.split(',')) > 1:
        net = nn.DataParallel(net).to(device)
    else :
        net = net.cuda()

    save_mode_path = os.path.join(
        snapshot_path, 'iter_10000_dice_0.7169.pth')

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    metric, dice_list,jacc_list, hd_list, ASD_list = test_all_case(net, val_loader =val_loader, val_files=val_files, method=args.model, num_classes=num_classes,
                               patch_size=args.patch_size, stride_xy=64, stride_z=64, save_result=True, test_save_path=test_save_path,
                               metric_detail=args.detail,nms=args.nms)

    return metric, dice_list,jacc_list, hd_list, ASD_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/data/sohui/Prostate/data/trim/ssl_data/centerCrop_200', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='SSL/MT_ATO_350_350_200_rampup_refpaper', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='Vnet_3D_256_randomCrop_30000_fold3', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--patch_size', type=list, default=[256,256,128],
                        help='patch size of network input')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='output channel of network')
    parser.add_argument('--gpu', type=str, default='4,5', help='GPU to use')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0,
                        help='apply NMS post-procssing?')
    parser.add_argument('--class_name', type=int, default=1)
    parser.add_argument('--fold', type=int, default=3, help='k fold cross validation')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric, dice_list,jacc_list, hd_list, ASD_list = Inference(args, device=device)
    for i in range((args.num_classes)-1):
        print('class:{}'.format(i+1))
        print('dice_mean:{}'.format(np.mean(dice_list[i])))
        #print('dice_std:{}'.format(np.std(dice_list[i])))
        print('jacc_mean:{}'.format(np.mean(jacc_list[i])))
        # print('jacc_std:{}'.format(np.std(jacc_list[i])))
        print('HD_mean:{}'.format(np.mean(hd_list[i])))
        #print('HD_std:{}'.format(np.std(hd_list[i])))
        print('ASD_mean:{}'.format(np.mean(ASD_list[i])))
        # print('ASD_std:{}'.format(np.std(ASD_list[i])))
