import glob
import os

import SimpleITK as sitk
import numpy as np

from settings import raw_folder, patch_folder


def convert_roi(roi_path):
    roi = np.zeros((2, 2))
    i = 0
    with open(roi_path) as f:
        for line in f.readlines():
            if line.startswith('point'):
                p = line.split('|')
                roi[i, 0] = -float(p[1])
                roi[i, 1] = -float(p[2])
                i += 1
    return roi


if __name__ == '__main__':
    tifs = sorted(glob.glob(raw_folder + '/*.tif'))
    tif_index = 0
    for tif_path in tifs:
        image = sitk.ReadImage(tif_path)
        spacing = image.GetSpacing()
        nda = sitk.GetArrayFromImage(image)
        tif_index += 1
        labels = sorted(glob.glob(raw_folder + '/' + os.path.basename(tif_path).split('.')[0] + '_label_*.nrrd'))
        label_index = 0
        for label_path in labels:
            label_index += 1
            label = sitk.ReadImage(label_path)
            label_nda = sitk.GetArrayFromImage(label).squeeze()
            roi_path = label_path.replace('.nrrd', '_roi.acsv')
            roi = convert_roi(roi_path)
            #
            center = list(roi[0])
            center = np.asarray(image.TransformPhysicalPointToIndex(center))
            length = np.abs(roi[1] / spacing).astype(np.int)
            start = center - length
            end = center + length
            #
            patch_nda = nda[start[1]:end[1], start[0]:end[0]]
            patch_label_nda = label_nda[start[1]:end[1], start[0]:end[0]]
            patch = sitk.GetImageFromArray(patch_nda)
            patch_label = sitk.GetImageFromArray(patch_label_nda)
            sitk.WriteImage(patch,
                            os.path.join(patch_folder, 'patch_' + str(tif_index) + '_' + str(label_index) + '.nrrd'))
            sitk.WriteImage(patch_label, os.path.join(patch_folder, 'patch_' + str(tif_index) + '_' + str(
                label_index) + '_label.nrrd'), True)
