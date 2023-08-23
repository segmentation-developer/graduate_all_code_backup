import cv2
import os
import numpy as np
import nibabel as nib

## raw image만 존재하는 dir
#data_path = '/home/psh/data/hanyang_Prostate/50_example/original/image'
##trim된 데이터의 image
data_path = '/home/psh/data/hanyang_Prostate/50_example/trim/image'
## PZ trim
label_path = '/home/psh/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class1_trim'
## TZ trim
#label_path = '/home/psh/data/hanyang_Prostate/50_example/trim/data_pre_before/label_nii_class2_trim'

for dirpath, dirnames, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirpath, filename))


data_file_list = sorted(os.listdir(data_path))
label_file_list = sorted(os.listdir(label_path))


for d in range(len(data_file_list)):

    ct_path = os.path.join(data_path, data_file_list[d])
    ct = nib.load(ct_path)
    affine = ct.affine
    ct = ct.get_fdata()

    mask_path = os.path.join(label_path, label_file_list[d])
    mask = nib.load(mask_path)
    # affine_mask = mask.affine
    mask = mask.get_fdata()

    ct = np.transpose(ct, (2, 0, 1))
    mask = np.transpose(mask, (2, 0, 1))

    # if affine_mask.any() != affine.any() :
    #    print('stop')

    # ct data slice
    ct = ct[:208, :, :]
    mask = mask[:208, :, :]

    c, w, h = ct.shape

    print('{},c:{}, w:{}, h:{}'.format(data_file_list[d], c, w, h))

    # if c < 208:
    #    z_padding = 208-c
    #    ct = np.pad(ct, ((0, z_padding), (0, 0), (0, 0)), 'constant')
    #    mask = np.pad(mask, ((0, z_padding), (0, 0), (0, 0)), 'constant')

    ## HU 조정 (-200 ~ 500)
    for i in range(len(ct)):
        ct[i] = np.where(ct[i] < -100, -100, ct[i])
        ct[i] = np.where(ct[i] > 200, 200, ct[i])

    ct = (ct - np.mean(ct)) / np.std(ct)  ##normalize
    ct = ct.astype(np.float32)
    print(ct.min(), ct.max())

    c_path = '/home/psh/data/hanyang_Prostate/50_example/image'
    m_path = '/home/psh/data/hanyang_Prostate/50_example/PZ_label/label_2_trim'

    if not os.path.exists(c_path):
        os.makedirs(c_path)
    if not os.path.exists(m_path):
        os.makedirs(m_path)

    c_path2 = c_path + '/{}'.format((data_file_list[d]))
    m_path2 = m_path + '/{}.nii.gz'.format((label_file_list[d].split('_')[1]))

    ct_t = np.transpose(ct, (1, 2, 0))
    mask_t = np.transpose(mask, (1, 2, 0))

    ct_t = nib.Nifti1Image(ct_t, affine)
    mask_t = nib.Nifti1Image(mask_t, affine)
    nib.save(ct_t, c_path2)
    nib.save(mask_t, m_path2)
