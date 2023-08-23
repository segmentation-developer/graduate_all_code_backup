
import nibabel as nib
#import matplotlib.pyplot as plt
# import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
#import tqdm as tqdm
from batchgenerators.utilities.file_and_folder_operations import save_json


task = 'unlabeled'
name = 'PZ'
data_path = '/home/psh/data/hanyang_Prostate/50_example/trim/ssl_data/ul_image'
## total prostate
if name == 'PZ' :
    label_path = '/home/psh/data/hanyang_Prostate/50_example/trim/sl_data/label_trim'
## transition zone
if name == 'TZ' :
    label_path = '/home/psh/data/hanyang_Prostate/50_example/trim/sl_data/label_2_trim'
label_file_list = sorted(os.listdir(label_path))

patient_names = []
train_patient_names = []
test_patient_names = []

if task == 'labeled' :
    file_lis = sorted(os.listdir(data_path))                # file_lis = data_file_list = 001.nii.gz~360.nii.gz

    for file_li in file_lis:
        image_file = os.path.join(data_path, file_li)       #image_file = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz
        #label_file = os.path.join(label_path, file_li)
        patient_names.append(image_file)            # patients_names = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz

    file_split = int(len(file_lis) * 0.8)
    train_patient_names = patient_names[:file_split]
    test_patient_names = patient_names[file_split:]


    json_dict = {}
    json_dict['name'] = "Prostate"
    #json_dict['description'] = "Gliomas segmentation tumour and oedema in on brain images"
    #json_dict['reference'] = "https://www.med.upenn.edu/sbia/brats2017.html"
    #json_dict['licence'] = "CC-BY-SA 4.0"
    #json_dict['release'] = "2.0 04/05/2018"
    json_dict['tensorImageSize'] = "4D"
    #json_dict['modality'] = {
    #     "0": "FLAIR",
    #	 "1": "T1w",
    #	 "2": "t1gd",
    #	 "3": "T2w"
    #}
    json_dict['labels'] = {
         "0": "background",
         "1": "prostate1"
         #"2": "prostate2"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)

    json_dict['training'] = [
        {'image': "./image/%s" % i.split("/")[-1], "label": "./label_2_trim/%s" % i.split("/")[-1]} for i in
        train_patient_names]

    json_dict['test'] = [
        {'image': "./image/%s" % i.split("/")[-1], "label": "./label_2_trim/%s" % i.split("/")[-1]} for i in
        test_patient_names]

    save_json(json_dict, os.path.join('/home/psh/data/hanyang_Prostate/50_example/trim', "dataset_2.json"))

else:
    file_lis = sorted(os.listdir(data_path))  # file_lis = data_file_list = 001.nii.gz~360.nii.gz

    for file_li in file_lis:
        image_file = os.path.join(data_path, file_li)  # image_file = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz
        # label_file = os.path.join(label_path, file_li)
        train_patient_names.append(image_file)  # patients_names = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz


    json_dict = {}
    json_dict['name'] = "Prostate"
    # json_dict['description'] = "Gliomas segmentation tumour and oedema in on brain images"
    # json_dict['reference'] = "https://www.med.upenn.edu/sbia/brats2017.html"
    # json_dict['licence'] = "CC-BY-SA 4.0"
    # json_dict['release'] = "2.0 04/05/2018"
    json_dict['tensorImageSize'] = "4D"
    # json_dict['modality'] = {
    #     "0": "FLAIR",
    #	 "1": "T1w",
    #	 "2": "t1gd",
    #	 "3": "T2w"
    # }
    json_dict['labels'] = {
        "0": "background",
        "1": "prostate1"
        # "2": "prostate2"
    }

    json_dict['numTraining'] = len(train_patient_names)

    json_dict['training'] = [
        {'image': "./ul_image/%s" % i.split("/")[-1]} for i in train_patient_names]


    save_json(json_dict, os.path.join('/home/psh/data/hanyang_Prostate/50_example/trim', "dataset_unlabeled.json"))