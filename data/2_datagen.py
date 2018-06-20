import glob
import os

import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import shutil

from data.helpers import flip_a_coin
from settings import npy_folder, patch_folder

DATA_TYPE = 'train'

np.random.seed(17)


def get_image_chunk(image, label, output_shape, chunk_size):
    # todo: we assume that both image and patch are squares, shape is also a single number
    input_shape = np.shape(label)[0]
    nda = np.zeros((chunk_size, shape, shape))
    nda_label = np.zeros_like(nda)
    for i in range(chunk_size):
        center = np.random.randint(int(output_shape / 2), int(input_shape - output_shape / 2), (1, 2))[0]
        image_patch = image[int(center[0] - output_shape / 2):int(center[0] + output_shape / 2),
                      int(center[1] - output_shape / 2):int(center[1] + output_shape / 2)]
        label_patch = label[int(center[0] - output_shape / 2):int(center[0] + output_shape / 2),
                      int(center[1] - output_shape / 2):int(center[1] + output_shape / 2)]
        if flip_a_coin():
            image_patch = np.fliplr(image_patch)
            label_patch = np.fliplr(label_patch)
        if flip_a_coin():
            image_patch = np.flipud(image_patch)
            label_patch = np.flipud(label_patch)
        k = np.random.randint(0, 4)
        image_patch = np.rot90(image_patch, k=k)
        label_patch = np.rot90(label_patch, k=k)
        nda[i] = image_patch
        nda_label[i] = label_patch
    return nda, nda_label


if __name__ == '__main__':
    labels = sorted(glob.glob(os.path.join(patch_folder, DATA_TYPE) + '/*_label.nrrd'))
    multiplier = 4000
    chunk_size = 8
    shape = 224
    output_folder = os.path.join(npy_folder, 'third_attempt')
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    else:
        shutil.rmtre(output_folder)
        os.mkdir(output_folder)
    for label_path in labels:
        image_path = label_path.replace('_label', '')
        input_image = sitk.ReadImage(image_path)
        input_label = sitk.ReadImage(label_path)
        input_image_nda = sitk.GetArrayFromImage(input_image)
        max_val = np.amax(input_image_nda)
        input_label_nda = sitk.GetArrayFromImage(input_label)
        nda = np.zeros((multiplier * chunk_size, shape, shape), dtype=np.float32)
        label_nda = np.zeros((multiplier * chunk_size, shape, shape), dtype=np.uint8)
        for i in tqdm(range(multiplier)):
            image_chunk, label_chunk = get_image_chunk(input_image_nda, input_label_nda, shape,
                                                       chunk_size)
            nda[i * chunk_size:(i + 1) * chunk_size, :, :] = image_chunk / max_val
            label_nda[i * chunk_size:(i + 1) * chunk_size, :, :] = label_chunk
        np.save(os.path.join(output_folder, os.path.basename(image_path).replace('nrrd', 'npy')),
                np.expand_dims(nda, -1))
        np.save(os.path.join(output_folder, os.path.basename(label_path).replace('nrrd', 'npy')),
                np.expand_dims(label_nda, -1))
