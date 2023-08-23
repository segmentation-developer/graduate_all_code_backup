import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
import os
import matplotlib.pyplot as plt
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
#import torchvision
#import albumentations

data_path = '/home/psh/data/Task05_Prostate/imagesTr'
label_path = '/home/psh/data/Task05_Prostate/labelsTr'
data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))

ivh = 0
ich = 0

for d in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[d+1])
    ct = nib.load(ct_path)
    affine = ct.affine
    ct = ct.get_fdata()
    ct = ct[:,:,:,0]


    mask_path = os.path.join(label_path,label_file_list[d+3])
    mask = nib.load(mask_path)
    mask = mask.get_fdata()


    '''
    plt.figure(figsize=(18, 18))
    # for idx in range(3):
    plt.subplot(3, 1, 1)
    plt.imshow(ct[:, :, 7:8])
    plt.subplot(3, 1, 3)
    plt.imshow(mask[:, :, 7:8])

    plt.tight_layout()
    plt.show()
    print()
    '''

    ct = np.transpose(ct,(2,0,1))
    mask = np.transpose(mask,(2,0,1))

    print(data_file_list[d+1])
    print(label_file_list[d+3])

    ith = []
    for i in range(mask.shape[0]):
        if mask[i].any() != 0.0:
           ith.append(i)
    print(ith)

    # ct data slice : label 이 존재하는 부분만 z_axis 자르기
    ct = ct[ith[0]:ith[-1], :320, :320]
    mask = mask[ith[0]:ith[-1], :320, :320]

    c, w, h = ct.shape
    if w < 320 or h < 320:
        print(data_file_list[d])

        w_padding = max((320 - w) // 2 + 3, 0)
        h_padding = max((320 - h) // 2 + 3, 0)
        ct = np.pad(ct, ((0, 0),(w_padding, w_padding), (h_padding, h_padding)), 'constant')
        mask = np.pad(mask, ((0, 0),(w_padding, w_padding), (h_padding, h_padding)), 'constant')
        print(ct.shape)

    '''
    if c < 250:
        z_padding = 250-c
        ct = np.pad(ct, ((0, z_padding), (0, 0), (0, 0)), 'constant')
        #mask = np.pad(mask, ((0, z_padding), (0, 0), (0, 0)), 'constant')
    '''
    ## HU 조정 (-200 ~ 500)
    for i in range(len(ct)):
        ct[i] = np.where(ct[i] < 0, 0, ct[i])
        ct[i] = np.where(ct[i] > 500, 500, ct[i])

    ##normalize
    ct = (ct - np.mean(ct)) / np.std(ct)
    ct = (ct - ct.min())/(ct.max()-ct.min())                 # 0~1
    ct = ct.astype(np.float32)
    print(ct.min(), ct.max())

    c_path = '/home/psh/data/Task05_Prostate/Task05_Prostate_pre2/imagesTr'
    m_path = '/home/psh/data/Task05_Prostate/Task05_Prostate_pre2/labelsTr'

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



    c_path2 = c_path + '/{}'.format((data_file_list[d+1]))
    m_path2 = m_path + '/{}'.format((label_file_list[d+3]))


    
    ct_t = np.transpose(ct, (1,2,0))
    mask_t = np.transpose(mask, (1,2,0))

    ct_t= nib.Nifti1Image(ct_t, affine)
    mask_t= nib.Nifti1Image(mask_t, affine)
    nib.save(ct_t, c_path2)
    nib.save(mask_t, m_path2)


