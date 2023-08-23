import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
#import torchvision
#import albumentations

data_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/images'
label_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/labels'
data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))

ivh = 0
ich = 0

for d in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[d])
    ct = nib.load(ct_path)
    affine = ct.affine
    ct = ct.get_fdata()


    mask_path = os.path.join(label_path,label_file_list[d])
    mask = nib.load(mask_path)
    mask = mask.get_fdata()

    ct = np.transpose(ct,(2,0,1))
    mask = np.transpose(mask,(2,0,1))


    ct = (ct - np.mean(ct)) / np.std(ct)                    ##normalize
    ct = (ct - ct.min())/(ct.max()-ct.min())                 # 0~1
    ct = ct.astype(np.float32)
    print(ct.min(), ct.max())

    c_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/images_norm'
    m_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/labels_norm'

        # print(m_path, i+1)
        # print(np.where(mask[i] == 1, True, False).sum())

        # ich_check = np.where(mask[i] == 1, True, False)
        # if ich_check.sum() != 0:
        #     ich += 1
        #
        # ivh_check = np.where(mask[i] == 2, True, False)
        # if ivh_check.sum() != 0:
        #     print(m_path, i+1)
        #     ivh += 1

    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(m_path):
        os.makedirs(m_path)



    c_path2 = c_path + '/{}'.format((data_file_list[d]))
    m_path2 = m_path + '/{}'.format((label_file_list[d]))


    '''
        c_path2 = c_path + '/{:03d}.png'.format(i + 1)
        m_path2 = m_path + '/m{:03d}.png'.format(i + 1) '''

    '''
        nib.save(ct[i], c_path2)
        nib.save(mask[i], m_path2)
    '''
    ct_t = np.transpose(ct, (1,2,0))
    mask_t = np.transpose(mask, (1,2,0))

    ct_t= nib.Nifti1Image(ct_t, affine)
    mask_t= nib.Nifti1Image(mask_t, affine)
    nib.save(ct_t, c_path2)
    nib.save(mask_t, m_path2)

        # cv2.imwrite(c_path2, ct[i])
        # cv2.imwrite(m_path2, mask[i])
'''
        # # 시각화해서 검증시
        plt.imsave(c_path2, ct[i])
        plt.imsave(m_path2, mask[i])
'''
