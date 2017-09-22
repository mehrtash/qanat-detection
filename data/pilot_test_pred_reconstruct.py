import glob
import numpy as np
import os
import SimpleITK as sitk
from settings import train_folder, raw_folder

if __name__ == '__main__':
    pred_path = os.path.join(train_folder, '2017_09_21_19_08_34', 'test_predictions', 'pilot.npy')
    if os.path.isfile(pred_path):
        pred = np.load(pred_path).squeeze(axis=-1)
        tif = sorted(glob.glob(raw_folder + '/*.tif'))[0]
        tif_image = sitk.ReadImage(tif)
        nda = sitk.GetArrayFromImage(tif_image)
        original_shape = nda.shape
        print(original_shape)
        nda = nda[2000:9000, :]
        nda = np.zeros_like(nda, dtype=np.float32)
        print(nda.shape)
        nda_binary = np.zeros_like(nda, dtype=np.uint8)
        patch_size = 224
        rangex = nda.shape[0] // patch_size
        rangey = nda.shape[1] // patch_size
        patches = np.zeros((rangex * rangey, patch_size, patch_size))
        print(patches.shape)
        index = 0
        for x in range(rangex):
            for y in range(rangey):
                bbox = (x * patch_size, y * patch_size, (x + 1) * patch_size, (y + 1) * patch_size)
                nda[bbox[0]:bbox[2], bbox[1]:bbox[3]] = pred[index]
                index += 1
        nda = np.lib.pad(nda, ((2000, original_shape[0]-9000), (0, 0)), 'constant')
        np.save(os.path.join(train_folder, '2017_09_21_19_08_34', 'test_predictions', 'pilot_reconstructed.npy'), nda)
        pred_image = sitk.GetImageFromArray(nda)
        pred_image.CopyInformation(tif_image)
        sitk.WriteImage(pred_image, os.path.join(train_folder, '2017_09_21_19_08_34',
                                                 'test_predictions', 'pilot_reconstructed.nrrd'))
