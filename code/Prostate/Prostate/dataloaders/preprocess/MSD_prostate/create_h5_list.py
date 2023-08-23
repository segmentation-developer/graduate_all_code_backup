
import nibabel as nib
import matplotlib.pyplot as plt
# import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
#from batchgenerators.utilities.file_and_folder_operations import save_json

data_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/images_norm_pad'
#label_path = '/data/sohui/Prostate/Prostate_nifiti_pre/label'
#data_file_list = sorted(os.listdir(data_path))
#label_file_list = sorted(os.listdir(label_path))

patient_names = []
train_patient_names = []
test_patient_names = []

file_lis = sorted(os.listdir(data_path))                # file_lis = data_file_list = 001.nii.gz~360.nii.gz

for file_li in file_lis:
    image_file = os.path.join(data_path, file_li)       #image_file = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz
    #label_file = os.path.join(label_path, file_li)
    patient_names.append(image_file)            # patients_names = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz

#file_split = int(len(file_lis) * 0.8)
train_patient_names = patient_names[:250]
val_patient_names = patient_names[250:275]
test_patient_names = patient_names[275:]

f = open("/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/images_norm_pad/train.txt",'w')
f__ = open("/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/images_norm_pad/val.txt",'w')
f_= open("/home/psh/data/Brats19_numclass_2/data/BraTs2019_nii/images_norm_pad/test.txt",'w')

for i in range(len(train_patient_names)):
    f.write(train_patient_names[i].split('/')[-1])
    if i != len(train_patient_names)-1:
        f.write('\n')
f.close()

for i in range(len(val_patient_names)):
    f__.write(val_patient_names[i].split('/')[-1])
    if i != len(val_patient_names)-1:
        f__.write('\n')
f__.close()

for i in range(len(test_patient_names)):
    f_.write(test_patient_names[i].split('/')[-1])
    if i != len(test_patient_names)-1:
        f_.write('\n')
f_.close()