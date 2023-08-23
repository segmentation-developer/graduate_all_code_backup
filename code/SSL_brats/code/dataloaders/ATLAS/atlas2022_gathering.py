import os
import nibabel as nib

nii_directory = '/home/psh/data/ATLAS_2/original'

train_folder =os.path.join(nii_directory,'train')
test_folder =os.path.join(nii_directory,'test')

trains_ = next(os.walk(train_folder))[1]
tests_ = next(os.walk(test_folder))[1]

trains_.sort()
tests_.sort()

##label
for train_ in trains_:
    train_folder_train_ = os.path.join(train_folder, train_)
    train__ = next(os.walk(train_folder_train_))[1]
    train__.sort()

    for i in (train__):
        file = os.path.join(train_folder_train_, i)
        folder = next(os.walk(file))[1]
        file = os.path.join(file , folder[0])
        folder = next(os.walk(file))[1]
        file = os.path.join(file, folder[0])

        word = next(os.walk(file))[2][0]
        if 'label' in word:                                  #label
            image_path = next(os.walk(file))[2][1]
            label_path = next(os.walk(file))[2][0]
        if 'T1w.nii.gz' in word:                              # image
            image_path = next(os.walk(file))[2][0]
            label_path = next(os.walk(file))[2][1]
        path = next(os.walk(file))[0]

        image_path = os.path.join(path, image_path)
        label_path = os.path.join(path, label_path)

        ##image save
        image = nib.load(image_path)
        affine = image.affine
        image = image.get_fdata()

        image_ = nib.Nifti1Image(image, affine)
        image_t_path='/home/psh/data/ATLAS_2/data/Label/images/' +'{}.nii.gz'.format(i)
        nib.save(image_, image_t_path)

        ##label save
        label = nib.load(label_path)
        label = label.get_fdata()

        label_ = nib.Nifti1Image(label, affine)
        label_t_path = '/home/psh/data/ATLAS_2/data/Label/labels/' + '{}.nii.gz'.format(i)
        nib.save(label_, label_t_path)

'''
##no label
for test_ in tests_:
    test_folder_test_ = os.path.join(test_folder, test_)
    test__ = next(os.walk(test_folder_test_))[1]
    test__.sort()

    for i in (test__):
        file = os.path.join(test_folder_test_, i)
        folder = next(os.walk(file))[1]
        file = os.path.join(file , folder[0])
        folder = next(os.walk(file))[1]
        file = os.path.join(file, folder[0])

        if next(os.walk(file))[2][0] == '*T1w.nii.gz':
            image_path = next(os.walk(file))[2][0]
        path = next(os.walk(file))[0]

        image_path = os.path.join(path, image_path)

        ##image save
        image = nib.load(image_path)
        affine = image.affine
        image = image.get_fdata()

        image_ = nib.Nifti1Image(image, affine)
        image_t_path='/home/psh/data/ATLAS_2/data/NoLabel/images/' +'{}.nii.gz'.format(i)
        nib.save(image_, image_t_path)

'''