import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
#import nrrd
import nibabel as nib
import os

output_size =[240,240,155]

def covert_h5():
    listt = glob('/home/psh/data/BraTS/Brats_nii_1251/*/*flair.nii.gz')
    for item in tqdm(listt):
        image = nib.load(item)
        image = image.get_fdata()
        label = nib.load(item.replace('flair.nii.gz', 'seg.nii.gz'))
        label = label.get_fdata()
        #label = (label == 255).astype(np.uint8)
        w, h, d = label.shape

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)

        path = ('/home/psh/data/BraTS/Brats_h5')
        path_ =  path +'/'+ item[-28:-13]
        os.makedirs(path_, exist_ok=True)

        f = h5py.File(path_+'/'+ item[-28:-13]+'.h5', 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()

if __name__ == '__main__':
    covert_h5()