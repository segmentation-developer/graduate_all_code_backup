import cv2
import numpy as np
import nibabel as nib

data_path = '/home/psh/data/Task05_Prostate/imagesTr'
label_path = '/home/psh/data/Task05_Prostate/labelsTr'

image = cv2.imread('example_image.jpg')