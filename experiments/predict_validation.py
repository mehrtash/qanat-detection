import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

module_root = '..'
sys.path.append(module_root)

from settings import intermediate_folder, train_folder
from model.predict import Segmenter

if __name__ == '__main__':

    train_folders = sorted(os.listdir(train_folder))
    train_folders.remove('board_logs')
    train_folders.remove('archive')
    for train_folder in train_folders[1:]:
        print('*%$'*30)
        print('*%$'*30)
        segmenter = Segmenter(intermediate_folder=intermediate_folder,
                              model_uid=train_folder)
        segmenter.predict_val()
