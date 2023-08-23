import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
#import torchvision
#import albumentations

data_path = '/home/psh/data/hanyang_Prostate/50_example'
#label_path = '/home/psh/data/Prostate/Prostate_nifiti/label'
data_file_list = sorted(os.listdir(data_path))




ivh = 0
ich = 0

for d in range(len(data_file_list)):

    anno1 = '/lesionAnnot3D-000.nii.gz'
    anno2 = '/lesionAnnot3D-001.nii.gz'
    ct1_path = os.path.join(data_path, data_file_list[d]+anno1)
    ct2_path = os.path.join(data_path, data_file_list[d] + anno2)
    ct1 = nib.load(ct1_path)
    ct2 = nib.load(ct2_path)
    affine1 = ct1.affine
    affine2 = ct2.affine

    if affine1.any() != affine2.any() :
        print('stop')
    ct1 = ct1.get_fdata()
    ct2 = ct2.get_fdata()

    if ct1.shape[0] != ct2.shape[0] or ct1.shape[1] != ct2.shape[1] or ct1.shape[2] != ct2.shape[2] :
        print('diff size')

    #ct1 = np.expand_dims(ct1, axis=0)
    #ct2 = np.expand_dims(ct2, axis=0)

    #label = np.concatenate((ct1, ct2), axis=0)

    label = np.where((ct2==255) ,2 , ct1)

    label = np.where((label == 255), 1, label)
    '''
    plt.figure(figsize=(18, 18))
    # for idx in range(3):
    plt.subplot(3, 1, 1)
    plt.imshow(ct1[:, :, 88:89])
    plt.subplot(3, 1, 2)
    plt.imshow(ct2[:, :, 88:89])
    plt.subplot(3, 1, 3)
    plt.imshow(label[:, :, 88:89])

    plt.tight_layout()
    plt.show()
    print()
    '''
    c_path = '/home/psh/data/hanyang_Prostate/50_example/label'


    if not os.path.exists(c_path):
        os.makedirs(c_path)

    c_path2 = c_path + '/{}.nii.gz'.format((data_file_list[d]))


    label= nib.Nifti1Image(label, affine1)
    nib.save(label, c_path2)
