import dicom2nifti
import os
import shutil

# dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)


# dicom_directory = '/home/has/Results/dcm_list/case_000/ANO_0087'
dicom_directory = '/home/psh/data/hanyang_Prostate/50_example/original/rawData_dcm_51-300/2'
# dicom_directory = '/home/has/Datasets/CV_CT/slice_5'
dicom_folder = next(os.walk(dicom_directory))[1]
dicom_folder.sort()
print(dicom_folder)

rst_path = '/home/psh/data/hanyang_Prostate/50_example/image_nifti_wholebody/201-250'
if os.path.isdir(rst_path) == False:
    os.makedirs(rst_path)

for case in dicom_folder:

    dicom_f1_n = next(os.walk(os.path.join(dicom_directory, case)))[1]
    dicom_f1 = os.path.join(dicom_directory, case, dicom_f1_n[0])


    input_path = dicom_f1
    #input_path = os.path.join (dicom_f2 , input_path [:])
    print(input_path)

    output_f_path = os.path.join(rst_path,'{}'.format((case)))
    print(output_f_path)
    if os.path.isdir(output_f_path)==False:
        os.makedirs(output_f_path)

    # output_path = os.path.join(output_f_path, 'imaging.nii.gz')
    # print(output_path)
    dicom2nifti.convert_directory(input_path, output_f_path, compression=True, reorient=False)
    # dicom2nifti.convert_directory(dicom_directory, output_path, compression=True, reorient=True)
    # icom2nifti.dicom_series_to_nifti(dicom_directory +'/'+dicom_list, output_path, reorient_nifti=True)

    output_before = next(os.walk(output_f_path))[2]
    output_before_path = os.path.join(output_f_path, output_before[0])
    # print(output_before_path)
    os.rename(output_before_path, os.path.join(rst_path,'{}.nii.gz'.format(case)))
    # print(output_f_path)
    print("")





# for dicom_list in dicom_folder:
#     print(dicom_list)
#     output_path = '/home/has/Datasets/(has)CT_nii/case_' + '{0:05d}'.format(dicom_folder.index(dicom_list))
#     os.makedirs(output_path)
#     input_path = dicom_directory + '/' + dicom_list
#     print(input_path)
#     dicom2nifti.convert_directory(input_path, output_path, compression=True, reorient=True)
#     # dicom2nifti.convert_directory(dicom_directory, output_path, compression=True, reorient=True)
#     # icom2nifti.dicom_series_to_nifti(dicom_directory +'/'+dicom_list, output_path, reorient_nifti=True)