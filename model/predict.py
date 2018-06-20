import json
import SimpleITK as sitk
from keras import backend as K
from skimage import io
import numpy as np
import os
import shutil
from tqdm import tqdm

module_root = '..'
import sys

sys.path.append(module_root)
from model.cnn import get_cnn
from data.helpers import shrink_image, correct_exposure, expand_image
from settings import patch_folder


class Segmenter(object):
    def __init__(self, intermediate_folder, model_uid):
        self.model_uid = model_uid
        self.intermediate_folder = intermediate_folder
        self.model_folder = os.path.join(intermediate_folder, 'train', model_uid)

    def __predict(self, vol_nda):
        cnn_config_file = os.path.join(self.model_folder, 'chronicle', 'model.json')
        with open(cnn_config_file) as json_file:
            config = json.load(json_file)
        cnn = get_cnn(config["id"], config["params"])
        model = cnn.model()
        model.load_weights(os.path.join(self.model_folder, 'model_checkpoint.hdf5'))
        predicted_prob = model.predict(vol_nda, verbose=1, batch_size=8)
        del model
        K.clear_session()
        return predicted_prob

    def predict_val(self, input_shape=224):
        print('predict validation')
        output_folder = os.path.join(self.model_folder, 'validation_predictions')
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        with open(os.path.join(self.model_folder, 'chronicle', 'dataset.json')) as json_file:
            dataset_config = json.load(json_file)
        data_folder = os.path.join(self.intermediate_folder, 'data', 'npy', dataset_config["npy uid"],
                                   str(dataset_config["folds"][0]))
        validation_file = os.path.join(data_folder, 'validation.txt')
        with open(validation_file) as f:
            validation_patches = f.read().splitlines()

        for patch in validation_patches:
            print(patch)
            patch_nrrd = os.path.join(patch_folder, patch.replace('_label.npy', '.nrrd'))
            # convert patch to npy
            patch_image = sitk.ReadImage(patch_nrrd)
            print(patch_image.GetSize())
            patch_nda = sitk.GetArrayFromImage(patch_image)
            max_val = np.amax(patch_nda)
            patch_nda = patch_nda / max_val
            patch_shape = patch_nda.shape
            step = 10
            n = (patch_shape[0] - input_shape) // step
            array = np.zeros((n ** 2, input_shape, input_shape, 1))
            counter = 0
            print(patch_nda.shape, array.shape, n)
            for i in range(n):
                for j in range(n):
                    array[counter, :, :, 0] = patch_nda[i * step:(i * step) + input_shape,
                                              j * step:j * step + input_shape]
                    counter += 1
            # predict
            prediction = self.__predict(array)
            # convert back npy to patch
            patch_pred = np.zeros_like(patch_nda)
            counter = 0
            for i in range(n):
                for j in range(n):
                    patch_pred[i * step:(i * step) + input_shape, j * step:j * step + input_shape] += prediction[counter, :,
                                                                                                      :, 0]
                    counter += 1
            patch_pred_image = sitk.GetImageFromArray(patch_pred)
            patch_pred_image.CopyInformation(srcImage=patch_image)

            # save prediction
            sitk.WriteImage(patch_pred_image,
                            os.path.join(output_folder, os.path.basename(patch_nrrd).replace('.nrrd', '_pred.nrrd')))
            # copy image
            shutil.copy(patch_nrrd, os.path.join(output_folder, os.path.basename(patch_nrrd)))
            # copy label
            shutil.copy(patch_nrrd.replace('.nrrd', '_label.nrrd'),
                        os.path.join(output_folder, os.path.basename(patch_nrrd).replace('.nrrd', '_label.nrrd')))

            # copy original with ground truth
        '''
        nda = np.load(os.path.join(data_folder, 'x_val.npy'))
        prediction = self.__predict(nda)
        np.save(os.path.join(output_folder, 'val_pred.npy'), prediction)
        '''

    def predict_test(self, name):
        output_folder = os.path.join(self.model_folder, 'test_predictions')
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        print('predict test')
        nda = np.load(os.path.join(self.intermediate_folder, 'data', 'npy', 'test', name + '.npy'))
        prediction = self.__predict(nda)
        np.save(os.path.join(output_folder, name + '.npy'), prediction)

    def predict_jpgs(self, image_paths, output_folder, shrink_factor, x_shape, rle=False):
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        for i, filename in enumerate(tqdm(image_paths)):
            image_original = io.imread(filename)
            image_corrected = correct_exposure(image_original)
            resized_x = shrink_image(image_corrected, shrink_factor=shrink_factor)
            pad0 = x_shape[0] - resized_x.shape[0]
            pad1 = x_shape[1] - resized_x.shape[1]
            padded_x = np.lib.pad(resized_x, ((int(pad0 / 2), int(pad0 / 2)),
                                              (int(pad1 / 2), int(pad1 / 2)),
                                              (0, 0)), mode='constant', constant_values=0)
            print(padded_x.shape)
            padded_x = np.expand_dims(padded_x, axis=0)
            prediction1 = self.__predict(padded_x)
            prediction2 = np.fliplr(self.__predict(np.fliplr(padded_x)))
            # prediction3 = np.flipud(self.__predict(np.flipud(padded_x)))
            # prediction = (prediction1 + prediction2 + prediction3) / 3.
            prediction = (prediction1 + prediction2) / 2.
            prediction = np.squeeze(prediction, axis=0)
            prediction = np.squeeze(prediction, axis=-1)
            prediction_shape = prediction.shape
            dx0 = int((prediction_shape[0] - resized_x.shape[0]) / 2)
            dx1 = int((prediction_shape[0] - resized_x.shape[0]) / 2)
            prediction = prediction[dx0:-dx0, dx1:-dx1]
            # padded_pred = np.lib.pad(prediction, ((0, 0),(int(pad1 / 2), int(pad1 / 2))), mode='constant', constant_values=0)
            pred_original = expand_image(prediction, expand_factor=shrink_factor)

            pred_original *= 255
            pred_original = pred_original.astype(np.uint8)
            output_path = os.path.join(output_folder, os.path.basename(filename).replace('.jpg', '_pred.png'))
            if rle:
                print('create rle')
            sitk.WriteImage(sitk.GetImageFromArray(pred_original), output_path)
