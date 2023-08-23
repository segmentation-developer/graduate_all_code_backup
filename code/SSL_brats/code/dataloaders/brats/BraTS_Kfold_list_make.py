
import nibabel as nib
import matplotlib.pyplot as plt
# import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
from batchgenerators.utilities.file_and_folder_operations import save_json
from sklearn.model_selection import KFold

data_path = '/home/psh/data/Brats19_numclass_2/data/BraTs2019'
#label_path = '/data/sohui/Prostate/Prostate_nifiti_pre/label'
data_file = sorted(os.listdir(data_path))
data_file_list = [file for file in data_file if file.endswith(".h5")]


#label_file_list = sorted(os.listdir(label_path))

patient_names = []


for file_li in data_file_list:
    image_file =  file_li      #image_file = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz
    #label_file = os.path.join(label_path, file_li)
    patient_names.append(image_file)            # patients_names = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz


kf = KFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kf.split(data_file_list)):
    train_patient_names = []
    test_patient_names = []
    for train in (train_idx):
        train_patient_names.append(patient_names[train])
    for val in (val_idx):
        test_patient_names.append(patient_names[val])

    f = open("/home/psh/data/Brats19_numclass_2/data/BraTs2019/fold{}/train.txt".format(fold+1), 'w')
    f_ = open("/home/psh/data/Brats19_numclass_2/data/BraTs2019/fold{}/test.txt".format(fold + 1), 'w')
    for i in range(len(train_patient_names)):
        f.write(train_patient_names[i])
        if i != len(train_patient_names) - 1:
            f.write('\n')
    f.close()

    for i in range(len(test_patient_names)):
        f_.write(test_patient_names[i])
        if i != len(test_patient_names) - 1:
            f_.write('\n')
    f_.close()