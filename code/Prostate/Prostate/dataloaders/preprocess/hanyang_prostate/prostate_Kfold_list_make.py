
import nibabel as nib
import matplotlib.pyplot as plt
# import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
from batchgenerators.utilities.file_and_folder_operations import save_json
from sklearn.model_selection import KFold

data_path = '/home/psh/data/hanyang_Prostate/50_example/trim/sl_data/centerCrop_350_350_200/image'
#label_path = '/data/sohui/Prostate/Prostate_nifiti_pre/label'
data_file_list = sorted(os.listdir(data_path))
#label_file_list = sorted(os.listdir(label_path))

patient_names = []



file_lis = sorted(os.listdir(data_path))                # file_lis = data_file_list = 001.nii.gz~360.nii.gz

for file_li in file_lis:
    image_file = os.path.join(data_path, file_li)       #image_file = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz
    #label_file = os.path.join(label_path, file_li)
    patient_names.append(image_file)            # patients_names = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz


kf = KFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(kf.split(file_lis)):
    train_patient_names = []
    test_patient_names = []
    for train in (train_idx):
        train_patient_names.append(patient_names[train])
    for val in (val_idx):
        test_patient_names.append(patient_names[val])


    json_dict = {}
    json_dict['name'] = "Prostate"
    #json_dict['description'] = "Gliomas segmentation tumour and oedema in on brain images"
    #json_dict['reference'] = "https://www.med.upenn.edu/sbia/brats2017.html"
    #json_dict['licence'] = "CC-BY-SA 4.0"
    #json_dict['release'] = "2.0 04/05/2018"
    json_dict['tensorImageSize'] = "2D"
    #json_dict['modality'] = {
    #     "0": "FLAIR",
    #	 "1": "T1w",
    #	 "2": "t1gd",
    #	 "3": "T2w"
    #}
    json_dict['labels'] = {
         "0": "background",
         "1": "PZ",
         "2": "TZ"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)

    json_dict['training'] = [
        {'image': "./image/%s" % i.split("/")[-1], "label": "./label_trim/%s" % i.split("/")[-1]} for i in
        train_patient_names]

    json_dict['test'] = [
        {'image': "./image/%s" % i.split("/")[-1], "label": "./label_trim/%s" % i.split("/")[-1]} for i in
        test_patient_names]

    save_json(json_dict, os.path.join('/home/psh/data/hanyang_Prostate/50_example/trim/sl_data/centerCrop_350_350_200', "dataset_fold{}.json".format(fold+1)))
