import glob
import os

import h5py
import numpy as np
import SimpleITK as sitk

'''
slice_num = 0
mask_path = sorted(glob.glob("/home/psh/data/Task05_Prostate/Task05_Prostate_pre_woMaskZeroSlices/imagesTr/*.nii.gz"))
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace("image", "label").replace(".nii.gz", ".nii.gz")
    if os.path.exists(msk_path):
        print(msk_path)
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)
        #image = (image - image.min()) / (image.max() - image.min())
        print(image.shape)
        image = image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0]
        if image.shape != mask.shape:
            print("Error")
        print(item)
        for slice_ind in range(image.shape[0]):
            f = h5py.File(
                '/home/psh/data/Task05_Prostate/Task05_Prostate_pre_slices2/label_data/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            f.close()
            slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))

'''

'''
slice_num = 0
mask_path = sorted(glob.glob("/home/psh/data/Task05_Prostate/Task05_Prostate_pre/imagesTs/*.nii.gz"))
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    #msk_path = case.replace("image", "label").replace(".nii.gz", ".nii.gz")
    if os.path.exists(case):
        #print(msk_path)
        #msk_itk = sitk.ReadImage(msk_path)
        #mask = sitk.GetArrayFromImage(msk_itk)
        #image = (image - image.min()) / (image.max() - image.min())
        print(image.shape)
        image = image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0]
        #if image.shape != mask.shape:
            #print("Error")
        print(item)
        for slice_ind in range(image.shape[0]):
            f = h5py.File(
                '/home/psh/data/Task05_Prostate/Task05_Prostate_pre_slices/unlabel_data/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            #f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            f.close()
            slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))
'''


############################################
# 기존 3D -> 2D img data에  Point_label 만들기
slice_num = 0
mask_path = sorted(glob.glob("/home/psh/data/Task05_Prostate/Task05_Prostate_pre_woMaskZeroSlices/imagesTr/*.nii.gz"))
for case in mask_path:
    img_itk = sitk.ReadImage(case)
    origin = img_itk.GetOrigin()
    spacing = img_itk.GetSpacing()
    direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)
    msk_path = case.replace("image", "label").replace(".nii.gz", ".nii.gz")
    if os.path.exists(msk_path):
        print(msk_path)
        msk_itk = sitk.ReadImage(msk_path)
        mask = sitk.GetArrayFromImage(msk_itk)
        #image = (image - image.min()) / (image.max() - image.min())
        print(image.shape)
        image = image.astype(np.float32)
        item = case.split("/")[-1].split(".")[0]
        if image.shape != mask.shape:
            print("Error")
        print(item)
        for slice_ind in range(image.shape[0]):
            f = h5py.File(
                '/home/psh/data/Task05_Prostate/Task05_Prostate_pre_slices2/label_data_wPointLabel/{}_slice_{}.h5'.format(item, slice_ind), 'w')
            f.create_dataset(
                'image', data=image[slice_ind], compression="gzip")
            f.create_dataset('label', data=mask[slice_ind], compression="gzip")
            f.close()
            slice_num += 1
print("Converted all ACDC volumes to 2D slices")
print("Total {} slices".format(slice_num))
