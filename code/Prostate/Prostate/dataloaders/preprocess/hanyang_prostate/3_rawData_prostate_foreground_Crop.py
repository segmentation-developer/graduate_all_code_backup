import cv2
import os
import numpy as np
import nibabel as nib
import monai
import torch


## img만 존재하는 raw data

## raw 데이터의 image
data_path = '/home/psh/data/hanyang_Prostate/50_example/image_nifti_wholebody/251-300'


data_file_list = sorted(os.listdir(data_path))

for k in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[k])
    ct = nib.load(ct_path)           # w,h,d
    affine = ct.affine
    ct = ct.get_fdata()


    ct = np.transpose(ct, (2, 0, 1))        # d,w,h


    # ct data z modi
    ct = ct[:250, :, :]

    d, w, h = int(ct.shape[0]),int(ct.shape[1]),int(ct.shape[2])

    print('{}, d:{}, w:{}, h:{}'.format(data_file_list[k], d, w, h))


    ## HU 조정 (-100 ~ 200)
    for i in range(len(ct)):
        ct[i] = np.where(ct[i] < -100, -100, ct[i])
        ct[i] = np.where(ct[i] > 200, 200, ct[i])

    ## normalization
    ct = (ct - ct.min()) / (ct.max() - ct.min())
    ## standard deviation
    #ct = (ct - np.mean(ct)) / np.std(ct)  ##normalize
    ct = ct.astype(np.float32)
    print(ct.min(), ct.max())


    ## crop foreground
    foreground_cropping = monai.transforms.CropForeground(margin=0)
    ct = foreground_cropping(ct)
    ct = ct.numpy()


    print('{}, d:{}, w:{}, h:{}'.format(data_file_list[k], ct.shape[0], ct.shape[1], ct.shape[2]))

    c_path = '/home/psh/data/hanyang_Prostate/50_example/trim/ssl_data/unlabeled/image_2'

    if not os.path.exists(c_path):
        os.makedirs(c_path)

    c_path2 = c_path + '/{}'.format((data_file_list[k]))

    ct_t = np.transpose(ct, (1, 2, 0))

    ct_t = nib.Nifti1Image(ct_t, affine)
    nib.save(ct_t, c_path2)
    print('__________________________________________')