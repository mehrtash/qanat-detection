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

    def predict_val(self):
        output_folder = os.path.join(self.model_folder, 'validation_predictions')
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        print('predict validation')
        with open(os.path.join(self.model_folder, 'chronicle', 'dataset.json')) as json_file:
            dataset_config = json.load(json_file)
        for angle in np.arange(1, 17, 1):
            angle_str = str(angle).zfill(2)
            data_folder = os.path.join(self.intermediate_folder, 'data', 'npy', dataset_config["npy uid"],
                                       str(dataset_config["folds"][0]))
            nda = np.load(os.path.join(data_folder, 'x_val_' + angle_str + '.npy'))
            print('predicting original...')
            prediction = self.__predict(nda)
            np.save(os.path.join(output_folder, 'val_pred_' + angle_str + '.npy'), prediction)
            print('predicting darker version...')
            nda *= 0.7
            prediction = self.__predict(nda)
            np.save(os.path.join(output_folder, 'val_pred_' + angle_str + '_d.npy'), prediction)
            print('predicting brighter version...')
            nda *= 2
            prediction = self.__predict(nda)
            nda[nda >= 1] = 1
            np.save(os.path.join(output_folder, 'val_pred_' + angle_str + '_brighter.npy'), prediction)

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
            dx0 = int((prediction_shape[0] - resized_x.shape[0])/2)
            dx1 = int((prediction_shape[0] - resized_x.shape[0])/2)
            prediction = prediction[dx0:-dx0,dx1:-dx1]
            # padded_pred = np.lib.pad(prediction, ((0, 0),(int(pad1 / 2), int(pad1 / 2))), mode='constant', constant_values=0)
            pred_original = expand_image(prediction, expand_factor=shrink_factor)

            pred_original *= 255
            pred_original = pred_original.astype(np.uint8)
            output_path = os.path.join(output_folder, os.path.basename(filename).replace('.jpg', '_pred.png'))
            if rle:
                print('create rle')
            sitk.WriteImage(sitk.GetImageFromArray(pred_original), output_path)




