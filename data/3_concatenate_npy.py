import numpy as np
import os
from settings import npy_folder
import glob
from utils.helpers import get_timestamp
from sklearn.model_selection import KFold

if __name__ == '__main__':
    tmp_folder = os.path.join(npy_folder, 'third_attempt')
    labels = glob.glob(tmp_folder + '/*_label.npy')
    uid = get_timestamp()
    output_folder = os.path.join(npy_folder, uid)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    kf = KFold(n_splits=7, shuffle=True)
    x = np.arange(0, len(labels))
    fold = 0
    for train_index, val_index in kf.split(x):
        print('fold: {}...'.format(fold))
        output_dir_fold = os.path.join(output_folder, str(fold))
        if not os.path.isdir(output_dir_fold):
            os.mkdir(output_dir_fold)
        train_label_files = sorted([labels[i] for i in train_index])
        val_label_files = sorted([labels[i] for i in val_index])
        fold += 1
        with open(os.path.join(output_dir_fold, 'training.txt'), 'w') as output_file:
            x_train_np_list = []
            y_train_np_list = []
            for filename in train_label_files:
                print('now reading: {}'.format(filename))
                output_file.write("%s\n" % os.path.basename(filename))
                x_train_np_list.append(np.load(filename.replace('_label', '')))
                y_train_np_list.append(np.load(filename))
            x_train = np.concatenate(x_train_np_list)
            y_train = np.concatenate(y_train_np_list)
            print('now saving...')
            np.save(os.path.join(output_dir_fold, 'x_train.npy'), x_train)
            np.save(os.path.join(output_dir_fold, 'y_train.npy'), y_train)
            del x_train, y_train

        with open(os.path.join(output_dir_fold, 'validation.txt'), 'w') as output_file:
            x_val_np_list = []
            y_val_np_list = []
            for filename in val_label_files:
                print('now reading: {}'.format(filename))
                output_file.write("%s\n" % os.path.basename(filename))
                x_val_np_list.append(np.load(filename.replace('_label', '')))
                y_val_np_list.append(np.load(filename))
            x_val = np.concatenate(x_val_np_list)
            y_val = np.concatenate(y_val_np_list)
            print('now saving...')
            np.save(os.path.join(output_dir_fold, 'x_val.npy'), x_val)
            np.save(os.path.join(output_dir_fold, 'y_val.npy'), y_val)
            del x_val, y_val

