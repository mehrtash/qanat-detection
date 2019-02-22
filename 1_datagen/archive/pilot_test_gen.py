import glob
import os

import SimpleITK as sitk
import numpy as np

from settings import raw_folder, npy_folder

if __name__ == '__main__':
    tifs = sorted(glob.glob(raw_folder + '/*.tif'))
    tif_index = 0
    for tif_path in tifs:
        img = sitk.ReadImage(tif_path)
        spacing = img.GetSpacing()
        nda = sitk.GetArrayFromImage(img)
        print(nda.shape)
        nda = nda[2000:9000, :]
        patch_size = 224
        rangex = nda.shape[0] // patch_size
        rangey = nda.shape[1] // patch_size
        patches = np.zeros((rangex*rangey, patch_size, patch_size))
        print(patches.shape)
        index = 0
        for x in range(rangex):
            for y in range(rangey):
                bbox = (x * patch_size, y * patch_size, (x+1) * patch_size, (y+1) * patch_size)
                patch = nda[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                max_val = np.amax(patch)
                # TODO: important fix this!!!
                patches[index] =  patch /max_val/max_val
                index += 1
        np.save(os.path.join(npy_folder, 'test', 'pilot.npy'), np.expand_dims(patches, axis=-1))

