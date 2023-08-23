import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
#import torchvision
#import albumentations


##trim된 데이터의 image
data_path = '/home/psh/data/hanyang_Prostate/50_example/trim/image'
label_path = '/home/psh/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class2_trim'
data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))

for d in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[d])
    ct = nib.load(ct_path)
    affine = ct.affine
    ct = ct.get_fdata()


    mask_path = os.path.join(label_path,label_file_list[d])
    mask = nib.load(mask_path)
    #affine_mask = mask.affine
    mask = mask.get_fdata()

    ct = np.transpose(ct,(2,0,1))
    mask = np.transpose(mask,(2,0,1))

    #if affine_mask.any() != affine.any() :
    #    print('stop')


    # ct data slice
    ct = ct[:208, :512, :512]
    mask = mask[:208, :512, :512]

    c, w, h = ct.shape
    

    if w!=512 or h!=512:
        print(data_file_list[d])

    #if c < 208:
    #    z_padding = 208-c
    #    ct = np.pad(ct, ((0, z_padding), (0, 0), (0, 0)), 'constant')
    #    mask = np.pad(mask, ((0, z_padding), (0, 0), (0, 0)), 'constant')

    ## HU 조정 (-200 ~ 500)
    for i in range(len(ct)):
        ct[i] = np.where(ct[i] < -100, -100, ct[i])
        ct[i] = np.where(ct[i] > 200, 200, ct[i])

    ct = (ct - np.mean(ct)) / np.std(ct)                    ##normalize
    #ct = (ct - ct.min())/(ct.max()-ct.min())                 # 0~1
    ct = ct.astype(np.float32)
    print(ct.min(), ct.max())

    c_path = '/home/psh/data/hanyang_Prostate/50_example/image'
    m_path = '/home/psh/data/hanyang_Prostate/50_example/PZ_label/label_2_trim'



    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(m_path):
        os.makedirs(m_path)



    c_path2 = c_path + '/{}'.format((data_file_list[d]))
    m_path2 = m_path + '/{}.nii.gz'.format((label_file_list[d].split('_')[1]))



    ct_t = np.transpose(ct, (1,2,0))
    mask_t = np.transpose(mask, (1,2,0))

    #ct_t= nib.Nifti1Image(ct_t, affine)
    mask_t= nib.Nifti1Image(mask_t, affine)
    #nib.save(ct_t, c_path2)
    nib.save(mask_t, m_path2)

'''
## image & label affine 맞춰서 먼저 저장하기

for d in range(len(data_file_list)):
    ct_path = os.path.join(data_path, data_file_list[d])
    ct = nib.load(ct_path)
    affine = ct.affine
    ct = ct.get_fdata()

    mask_path = os.path.join(label_path, label_file_list[d])
    mask = nib.load(mask_path)
    mask = mask.get_fdata()

    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(m_path):
        os.makedirs(m_path)


    c_path2 = c_path + '/{}'.format((data_file_list[d]))
    m_path2 = m_path + '/{}'.format((label_file_list[d]))



    ct_t= nib.Nifti1Image(ct_t, affine)
    mask_t= nib.Nifti1Image(mask_t, affine)
    nib.save(ct_t, c_path2)
    nib.save(mask_t, m_path2)
'''