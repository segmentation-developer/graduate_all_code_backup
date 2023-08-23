import cv2
import os
import numpy as np
import nibabel as nib
import monai
import torch


##trim된 데이터의 image
data_path = '/home/psh/data/hanyang_Prostate/50_example/image_nifti_wholebody/1-50'
## PZ trim
label_path = '/home/psh/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class1_trim'
## TZ trim
#label_path = '/home/psh/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class2_trim'



data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))


for k in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[k])
    ct = nib.load(ct_path)           # w,h,d
    affine = ct.affine
    ct = ct.get_fdata()

    mask_path = os.path.join(label_path, label_file_list[k])
    mask = nib.load(mask_path)      # w,h,d
    # affine_mask = mask.affine
    mask = mask.get_fdata()

    ## img & label 한 쌍인지 확인
    ct_name = ct_path.split('/')[-1].split('.')[0]
    mask_name = mask_path.split('/')[-1].split('_')[1]
    if ct_name != mask_name :
        print(ct_path)
        continue

    ct = np.transpose(ct, (2, 0, 1))        # d,w,h
    mask = np.transpose(mask, (2, 0, 1))        # d,w,h

    # if affine_mask.any() != affine.any() :
    #    print('stop')

    # ct data z modi
    ct = ct[:250, :, :]
    mask = mask[:250, :, :]

    d, w, h = int(ct.shape[0]),int(ct.shape[1]),int(ct.shape[2])

    print('{}, d:{}, w:{}, h:{}'.format(data_file_list[k], d, w, h))

    # if c < 208:
    #    z_padding = 208-c
    #    ct = np.pad(ct, ((0, z_padding), (0, 0), (0, 0)), 'constant')
    #    mask = np.pad(mask, ((0, z_padding), (0, 0), (0, 0)), 'constant')

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

    c_path = '/home/psh/data/hanyang_Prostate/50_example/trim/ssl_data/image'
    m_path = '/home/psh/data/hanyang_Prostate/50_example/trim/ssl_data/label_trim'

    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(m_path):
        os.makedirs(m_path)

    c_path2 = c_path + '/{}'.format((data_file_list[k]))
    m_path2 = m_path + '/{}.nii.gz'.format((label_file_list[k].split('_')[1]))

    ct_t = np.transpose(ct, (1, 2, 0))
    mask_t = np.transpose(mask, (1, 2, 0))

    ct_t = nib.Nifti1Image(ct_t, affine)
    mask_t = nib.Nifti1Image(mask_t, affine)
    nib.save(ct_t, c_path2)
    nib.save(mask_t, m_path2)
    print('__________________________________________')


## img만 존재하는 raw data

