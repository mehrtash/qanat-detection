import json
import os
import numpy as np


module_root = '..'
import sys
sys.path.append(module_root)

from utils.helpers import get_timestamp
from model.cnn import get_cnn


class Experiments:
    """
    Experiments to train a network. To run move the tank.
    """
    def __init__(self, json_file, intermediate_folder):
        jfile = open(json_file)
        config = json.load(jfile)
        jfile.close()
        self.experiment = config["experiment"]
        self.intermediate_folder = intermediate_folder
        self.train_folder = os.path.join(self.intermediate_folder, 'train')

    def run(self):
        datasets = self.experiment["datasets"]
        cnn_params = self.experiment["cnns"]
        trainparams = self.experiment["trainparams"]
        for dataset in datasets:
            for fold in dataset["folds"]:
                for cnn_param in cnn_params:
                    for trainparam in trainparams:
                        self.train(cnn_params=cnn_param, dataset=dataset, fold=fold, train_params=trainparam)

    def data(self, dataset, chronicle_folder):
        npy_uid = dataset["npy uid"]
        fold = dataset["folds"][0]
        mean_sub = dataset["mean sub"]
        std_div = dataset["std div"]
        data_folder = os.path.join(self.intermediate_folder, 'data', 'npy', npy_uid, str(fold))
        x = np.load(os.path.join(data_folder,  'x_train.npy'))
        y = np.load(os.path.join(data_folder, 'y_train.npy'))
        x_val = np.load(os.path.join(data_folder, 'x_val.npy'))
        y_val = np.load(os.path.join(data_folder, 'y_val.npy'))
        if mean_sub:
            print("mean_sub")
        if std_div:
            print("std_div")
        data = (x, y, (x_val, y_val))
        return data

    def train(self, cnn_params, dataset, fold, train_params):

        uid = get_timestamp()
        experiment_folder = os.path.join(self.train_folder, uid)
        os.mkdir(experiment_folder)
        chronicle_folder = os.path.join(experiment_folder, 'chronicle')
        os.mkdir(chronicle_folder)
        dataset["folds"] = [fold, ]

        # chroniclogy
        # pd.DataFrame.from_dict(self.experiment, orient='index').to_json(os.path.join(chronicle_folder, 'experiment.json'))
        with open(os.path.join(chronicle_folder, 'experiment.json'), 'w') as output_json_file:
            json.dump(self.experiment, output_json_file)
        # pd.DataFrame.from_dict(dataset, orient='index').to_json(os.path.join(chronicle_folder, 'dataset.json'))
        with open(os.path.join(chronicle_folder, 'dataset.json'), 'w') as output_json_file:
            json.dump(dataset, output_json_file)
        # pd.DataFrame.from_dict(cnn_params, orient='index').to_json(os.path.join(chronicle_folder, 'model.json'))
        with open(os.path.join(chronicle_folder, 'model.json'), 'w') as output_json_file:
            json.dump(cnn_params, output_json_file)

        # pd.DataFrame.from_dict(train_params, orient='index').to_json(os.path.join(chronicle_folder, 'train.json'))
        with open(os.path.join(chronicle_folder, 'train.json'), 'w') as output_json_file:
            json.dump(train_params, output_json_file)

        cnn = get_cnn(cnn_params["id"], cnn_params["params"])

        from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, ReduceLROnPlateau, EarlyStopping
        from keras.utils import plot_model
        model = cnn.model()
        plot_model(model, to_file=os.path.join(experiment_folder, 'model.png'), show_shapes=True)
        with open(os.path.join(experiment_folder, 'summary.txt'), 'w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        print('loading data...')
        data = self.data(dataset, chronicle_folder)

        print('-' * 100)
        x, y, validation_data = data

        print('-' * 30)
        print('Creating and compiling model...')
        print('-' * 30)
        csv_logger = CSVLogger(os.path.join(experiment_folder,  'log.csv'))
        checkpoint = train_params["checkpoint"]
        model_checkpoint = ModelCheckpoint(os.path.join(experiment_folder, 'model_checkpoint.hdf5'),
                                           monitor=checkpoint["monitor"],
                                           mode=checkpoint["mode"],
                                           save_best_only=True)
        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        lr = train_params["lr"]
        callbacks_list = [model_checkpoint, csv_logger]
        if lr["policy"] == "reduce lr":
            params = lr["params"]
            reduce_lr = ReduceLROnPlateau(monitor=params["monitor"],
                                          mode=params["mode"],
                                          factor=params["factor"],
                                          patience=params["patience"],
                                          min_lr=params["min_lr"],
                                          epsilon=params["epsilon"],
                                          verbose=1)
            callbacks_list.append(reduce_lr)

        if train_params["early stopping"]:
            es = EarlyStopping(monitor='val_dice_coef', mode='max', min_delta=1e-5, patience=60, verbose=1)
            callbacks_list.append(es)
        board = TensorBoard(log_dir=os.path.join(self.train_folder, 'board_logs'), histogram_freq=0, write_graph=True)
        # usage: tensorboard --logdir=/full_path_to_your_logs
        callbacks_list.append(board)

        with open(os.path.join(experiment_folder, 'model.json'), 'w') as outfile:
            json.dump(model.to_json(), outfile)

        if bool(cnn_params["summary"]):
            print(model.summary())
        model.fit(x, y, batch_size=train_params["batch size"],
                      epochs=train_params["epochs"],
                      verbose=train_params["verbose"],
                      shuffle=True,
                      callbacks=callbacks_list,
                      validation_data=validation_data)

