
import nibabel as nib
import matplotlib.pyplot as plt
# import cv2
import os
import numpy as np
import tqdm as tqdm

##Label directory
label_data_path = '/home/psh/data/ATLAS_2/data/Label/images'        #655

patient_names = []


label_file_lis = sorted(os.listdir(label_data_path))

for file_li in label_file_lis:
    image_file = os.path.join(label_data_path, file_li)
    patient_names.append(image_file)



train_label_patient_names = patient_names[:68]
train_unlabel_patient_names = patient_names[68:388]
val_patient_names = patient_names[388:464]
test_patient_names = patient_names[464:]

f = open("/home/psh/data/ATLAS_2/data/Label/train_label.txt", 'w')
for i in train_label_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()

f = open("/home/psh/data/ATLAS_2/data/Label/train_unlabel.txt", 'w')
for i in train_unlabel_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()

f = open("/home/psh/data/ATLAS_2/data/Label/val.txt", 'w')
for i in val_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()

f = open("/home/psh/data/ATLAS_2/data/Label/test.txt", 'w')
for i in test_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()



##NoLabel directory

unlabel_data_path = '/home/psh/data/ATLAS_2/data/NoLabel/images'

u_patient_names = []

file_lis = sorted(os.listdir(unlabel_data_path))

for file_li in file_lis:
    image_file = os.path.join(unlabel_data_path, file_li)
    u_patient_names.append(image_file)


f = open("/home/psh/data/ATLAS_2/data/Label/train_unlabel.txt", 'a')
for i in u_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()



##SL data split
'''
train_label_patient_names = patient_names[:472]
val_patient_names = patient_names[472:524]
test_patient_names = patient_names[524:]

f = open("/home/psh/data/ATLAS_2/data/Label/SL/train.txt", 'w')
for i in train_label_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()


f = open("/home/psh/data/ATLAS_2/data/Label/SL/val.txt", 'w')
for i in val_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()

f = open("/home/psh/data/ATLAS_2/data/Label/SL/test.txt", 'w')
for i in test_patient_names:
    data = i.split("/")[-1]
    f.write(data)
    f.write('\n')
f.close()
'''