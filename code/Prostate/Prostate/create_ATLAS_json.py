
import nibabel as nib
import matplotlib.pyplot as plt
# import cv2
import os
# https://www.kaggle.com/kmader/show-3d-nifti-images
import numpy as np
import tqdm as tqdm
from batchgenerators.utilities.file_and_folder_operations import save_json

data_path = '/home/psh/data/ATLAS_2/data/Label/images'          ##GT가 있는 images(655)
unlabel_data_path = '/home/psh/data/ATLAS_2/data/NoLabel/images'    ##GT가 없는 images(300)

patient_names = []
train_patient_names = []
test_patient_names = []

file_lis = sorted(os.listdir(data_path))                # file_lis = data_file_list = 001.nii.gz~360.nii.gz

for file_li in file_lis:
    image_file = os.path.join(data_path, file_li)       #image_file = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz
    label_file = os.path.join(data_path, file_li)
    patient_names.append(image_file)            # patients_names = /data3/sohui/hanyang_brain/3d_modi/data/001.nii.gz


train_patient_names = patient_names[:file_split]
test_patient_names = patient_names[file_split:]


json_dict = {}
json_dict['name'] = "BRATS"
json_dict['description'] = "Gliomas segmentation tumour and oedema in on brain images"
json_dict['reference'] = "https://www.med.upenn.edu/sbia/brats2017.html"
json_dict['licence'] = "CC-BY-SA 4.0"
json_dict['release'] = "2.0 04/05/2018"
json_dict['tensorImageSize'] = "4D"
json_dict['modality'] = {
     "0": "FLAIR",
	 "1": "T1w",
	 "2": "t1gd",
	 "3": "T2w"
}
json_dict['labels'] = {
     "0": "background",
	 "1": "edema",
	 "2": "non-enhancing tumor",
	 "3": "enhancing tumour"
}

json_dict['numTraining'] = len(train_patient_names)
json_dict['numTest'] = len(test_patient_names)

json_dict['training'] = [
    {'image': "./data/%s" % i.split("/")[-1], "label": "./label/m%s" % i.split("/")[-1]} for i in
    train_patient_names]

json_dict['train_validation_test'] = [
    {'image': "./data/%s" % i.split("/")[-1], "label": "./label/m%s" % i.split("/")[-1]} for i in
    test_patient_names]

save_json(json_dict, os.path.join('/data/sohui/hanyang_brain/nii_hBrain', "dataset.json"))