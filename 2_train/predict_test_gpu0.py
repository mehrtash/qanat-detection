import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

module_root = '..'
sys.path.append(module_root)

from settings import intermediate_folder, train_folder, raw_folder
from model.predict import Segmenter

if __name__ == '__main__':

    train_folders = os.listdir(train_folder)
    train_folders.remove('board_logs')
    # train_folders = ['', ]
    for train_folder in train_folders:
        segmenter = Segmenter(intermediate_folder=intermediate_folder, model_uid=train_folder)
        segmenter.predict_test('pilot')
