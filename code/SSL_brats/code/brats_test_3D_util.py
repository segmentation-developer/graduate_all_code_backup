import math

import numpy
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm
import matplotlib.pyplot as plt

def getLargestCC(segmentation):
    labels = label(segmentation)

    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    return largestCC

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    #y1 = net(test_patch)
                    y1,_ = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]

    return score_map,label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)


def test_all_case(net, base_dir, method="unet_3D", test_list="test_list.txt", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, save_result=True, test_save_path=None, metric_detail=0 , nms=0, fold=None):
    with open(base_dir + '/fold{}/{}'.format(fold,test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}".format(item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 4))
    total_uncertainty = 0
    print("Testing begin")
    with open(test_save_path + "/{}.txt".format(method), "a") as f:
        ith = 0
        loader = tqdm(image_list) if not metric_detail else image_list
        for image_path in loader:
            name = image_path[-15:-3]
            ids = image_path.split("/")[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]

            score_map,prediction = test_single_case(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            uncertainty_scoreMap = -1.0 * numpy.sum(score_map * numpy.log(score_map + 1e-6), axis=0)
            uncertainty_mean = numpy.mean(uncertainty_scoreMap)
            total_uncertainty += uncertainty_mean
            if nms:
                prediction = getLargestCC(prediction)

            metric = calculate_metric_percase(label == 1, prediction == 1)
            total_metric[0, :] += metric
            f.writelines("{},{},{},{},{}, uncertainty:{}\n".format(
                ith, metric[0], metric[1], metric[2], metric[3],uncertainty_mean))

            if metric_detail:
                print('%02d,\t%.5f, %.5f, %.5f, %.5f uncertainty:%.5f' % (
                    ith , metric[0], metric[1], metric[2], metric[3], uncertainty_mean ))

            if save_result:
                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, test_save_path +
                                "/{}_{}_pred.nii.gz".format(ith,name))
                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, test_save_path +
                                "/{}_{}_img.nii.gz".format(ith,name))

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, test_save_path +
                                "/{}_{}_gt.nii.gz".format(ith,name))
                ##uncertainty map
                uncertainty_itk = sitk.GetImageFromArray(uncertainty_scoreMap)
                uncertainty_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(uncertainty_itk, test_save_path +
                                "/{}_{}_uncertainty.nii.gz".format(ith, name))

            ith += 1

        f.writelines("Mean metrics,{},{},{},{},{}".format(total_metric[0, 0] / len(image_list), total_metric[0, 1] / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list),total_uncertainty/ len(image_list) ))
    f.close()
    print("Testing end")
    return total_metric / len(image_list), total_uncertainty/ len(image_list)


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    #ravd = abs(metric.binary.ravd(pred, gt))
    if dice == 0 :
        hd = 0
        asd = 0
    else :
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
    return np.array([dice, jc, hd, asd])
