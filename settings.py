import os
import socket

folders = []
if socket.gethostname() in ['bezd']:
    root = "/home/mehrtash/dropbox/qanat_v2/"
    raw_folder = os.path.join(root, 'raw')
    intermediate_folder = os.path.join(root, 'intermediate')

folders.append(raw_folder)
folders.append(intermediate_folder)
# data folders
data_folder = os.path.join(intermediate_folder, 'data')
folders.append(data_folder)
patch_folder = os.path.join(data_folder, 'patch', 'train')
folders.append(patch_folder)
split_folder = os.path.join(data_folder, 'split')
folders.append(split_folder)
sheets_folder = os.path.join(intermediate_folder, 'sheets')
folders.append(sheets_folder)
ccf_folder = os.path.join(data_folder, 'ccf')
folders.append(ccf_folder)
npy_folder = os.path.join(data_folder, 'npy')
folders.append(npy_folder)
# model folders
model_folder = os.path.join(intermediate_folder, 'model')
folders.append(model_folder)
model_checkpoints_folder = os.path.join(model_folder, 'checkpoints')
folders.append(model_checkpoints_folder)
# model folders
train_folder = os.path.join(intermediate_folder, 'train')
folders.append(train_folder)

if __name__ == '__main__':
    for folder in folders:
        if not os.path.isdir(folder):
            os.mkdir(folder)
