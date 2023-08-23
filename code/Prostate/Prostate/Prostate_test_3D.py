import argparse
import os
import shutil
import numpy as np
import torch
from Prostate.networks.unet_3D import unet_3D
#from networks.vnet import VNet
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
            Orientationd(keys=["image", "label"], axcodes="LPS"),   #LPS ->
            Spacingd(
                keys=["image", "label"],
                pixdim=(0.8,0.8,0.8),
                mode=("bilinear", "nearest"),
            ),
            CenterSpatialCropd(keys=['image', 'label'], roi_size=(176,176,176)),
            #SpatialPadd(keys=["image", "label"], spatial_size=(320, 320, 32), mode="constant"),
        ]
    )

    datasets = args.root_path + "/dataset.json"
    val_files = load_decathlon_datalist(datasets, True, "test")

    db_val = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=4
    )
    val_loader = DataLoader(
        db_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    snapshot_path = "/data/sohui/Prostate/{}/{}_{}".format(args.exp, args.model,args.max_iterations)
    #snapshot_path = "/data/sohui/MSD_prostate/{}/{}".format(args.exp, args.model)
    num_classes = args.num_classes
    #test_save_path = "/data/sohui/LA_dataset/{}/{}".format(FLAGS.exp_test,FLAGS.denseUnet_3D)
    test_save_path = "/data/sohui/Prostate/prostate_test_result/{}/{}_{}".format(args.exp, args.model,args.max_iterations)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = unet_3D(n_classes=num_classes, in_channels=1)

    if len(args.gpu.split(',')) > 1:
        net = nn.DataParallel(net).to(device)
    else :
        net = net.cuda()

    save_mode_path = os.path.join(
        snapshot_path, 'model_iter_2200_dice_0.846.pth')

    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()
    metric, dice_list,jacc_list, hd_list, ASD_list = test_all_case(net, val_loader =val_loader, val_files=val_files, method=args.model, num_classes=num_classes,
                               patch_size=args.patch_size, stride_xy=32, stride_z=32, save_result=True, test_save_path=test_save_path,
                               metric_detail=args.detail,nms=args.nms)

    return metric, dice_list,jacc_list, hd_list, ASD_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default='/data/sohui/Prostate', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='unet_slidingW', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='unet_3D_192', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=10000, help='maximum epoch number to train')
    parser.add_argument('--patch_size', type=list, default=[128,128,176],
                        help='patch size of network input')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='output channel of network')
    parser.add_argument('--gpu', type=str, default='6,7', help='GPU to use')
    parser.add_argument('--detail', type=int, default=1,
                        help='print metrics for every samples?')
    parser.add_argument('--nms', type=int, default=0,
                        help='apply NMS post-procssing?')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric, dice_list,jacc_list, hd_list, ASD_list = Inference(args, device=device)
    #print(metric)
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
