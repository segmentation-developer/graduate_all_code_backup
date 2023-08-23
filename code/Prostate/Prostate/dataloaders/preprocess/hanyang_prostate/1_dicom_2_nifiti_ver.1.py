import os
import glob
import pydicom
import numpy as np
import nibabel as nib

# dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)


# dicom_directory = '/home/has/Results/dcm_list/case_000/ANO_0087'
dicom_directory = '/home/psh/data/hanyang_Prostate/50_example/original/51-100_CT_raw_data'
# dicom_directory = '/home/has/Datasets/CV_CT/slice_5'
dicom_folder = next(os.walk(dicom_directory))[1]
dicom_folder.sort()
print(dicom_folder)

rst_path = '/home/psh/data/hanyang_Prostate/50_example/image'
if os.path.isdir(rst_path) == False:
    os.makedirs(rst_path)

for case in dicom_folder:

    dicom_f1_n = next(os.walk(os.path.join(dicom_directory, case)))[1]
    # dicom file이 있는 경로
    dicom_f1 = os.path.join(dicom_directory, case, dicom_f1_n[0])
    # DICOM 파일 목록 얻기
    dicom_files = glob.glob(os.path.join(dicom_f1, "*.dcm"))

    # DICOM 파일들을 읽어들여 데이터와 헤더 정보 추출
    dicom_data = []
    dicom_headers = []
    for dicom_file in dicom_files:
        dicom = pydicom.read_file(dicom_file)
        dicom_data.append(dicom.pixel_array)
        dicom_headers.append(dicom)

    # DICOM 파일들의 헤더 정보 비교
    assert all(header.PixelSpacing == dicom_headers[0].PixelSpacing for header in dicom_headers)
    assert all(header.SliceThickness == dicom_headers[0].SliceThickness for header in dicom_headers)

    # DICOM 데이터를 4D 배열로 변환
    dicom_data = np.array(dicom_data)

    # DICOM 파일의 변환 행렬과 pixel spacing 정보 추출
    affine = []
    affine[0, 0] = dicom_headers[0].PixelSpacing[0]
    affine[1, 1] = dicom_headers[0].PixelSpacing[1]
    affine[2, 2] = dicom_headers[0].SliceThickness

    # DICOM 데이터를 NIfTI 형식으로 변환하여 저장
    nifti_img = nib.Nifti1Image(dicom_data, affine)
    nib.save(nifti_img,os.path.join(rst_path, '{}.nii.gz'.format(case)))




# for dicom_list in dicom_folder:
#     print(dicom_list)
#     output_path = '/home/has/Datasets/(has)CT_nii/case_' + '{0:05d}'.format(dicom_folder.index(dicom_list))
#     os.makedirs(output_path)
#     input_path = dicom_directory + '/' + dicom_list
#     print(input_path)
#     dicom2nifti.convert_directory(input_path, output_path, compression=True, reorient=True)
#     # dicom2nifti.convert_directory(dicom_directory, output_path, compression=True, reorient=True)
#     # icom2nifti.dicom_series_to_nifti(dicom_directory +'/'+dicom_list, output_path, reorient_nifti=True)