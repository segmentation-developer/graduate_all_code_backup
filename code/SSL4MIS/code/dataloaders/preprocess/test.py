import nibabel as nib
import matplotlib.pyplot as plt
#import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
#import torchvision
#import albumentations
import h5py

data_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019'
data_file_list = sorted(os.listdir(data_path))

ivh = 0
ich = 0

for d in (data_file_list):

    h5f = h5py.File(data_path + "/{}".format(d), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]



    image = (image - np.mean(image)) / np.std(image)                    ##normalize
    image = (image - image.min())/(image.max()- image.min())
    image = image.astype(np.float32)
    print(image.min(), image.max())


    c_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019_norm'

    if not os.path.exists(c_path):
        os.makedirs(c_path)


    c_path2 = c_path + '/{}'.format(d)

    f = h5py.File(c_path2, 'w')
    f.create_dataset('image', data=image, compression="gzip")
    f.create_dataset('label', data=label, compression="gzip")
    f.close()