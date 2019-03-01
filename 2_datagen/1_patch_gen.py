import glob
import os

import SimpleITK as sitk
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

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
    tifs = sorted(glob.iglob(os.path.join(raw_folder, 'third_attempt') + '/**/*.tif', recursive=True))
    tif_index = 0
    output_train_folder = os.path.join(patch_folder, 'train')
    if not os.path.isdir(output_train_folder):
        os.mkdir(output_train_folder)
    output_test_folder = os.path.join(patch_folder, 'test')
    if not os.path.isdir(output_test_folder):
        os.mkdir(output_test_folder)
    d = list()
    for tif_path in tqdm(tifs):
        if 'train' in tif_path:
            output_folder = output_train_folder
            patch_type = 'train'
        else:
            output_folder = output_test_folder
            patch_type = 'test'
        image = sitk.ReadImage(tif_path)
        spacing = image.GetSpacing()
        nda = sitk.GetArrayFromImage(image)
        tif_index += 1
        labels = sorted(
            glob.glob(os.path.dirname(tif_path) + '/' + os.path.basename(tif_path).split('.')[0] + '-label_*.nrrd'))
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
                            os.path.join(output_folder, 'patch_' + str(tif_index) + '_' + str(label_index) + '.nrrd'))
            sitk.WriteImage(patch_label, os.path.join(output_folder, 'patch_' + str(tif_index) + '_' + str(
                label_index) + '_label.nrrd'), True)
            d.append(OrderedDict(
                {'patch_type': patch_type, 'patch_id': str(tif_index) + '_' + str(label_index), 'tif': tif_path,
                 'label': label_path}))
    pd.DataFrame(d).to_csv(os.path.join(patch_folder, 'patch_ids.csv'))
