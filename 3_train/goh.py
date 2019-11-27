import os
import sys
sys.path.append('..')

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model.cnns.unet import Unet

if __name__ == '__main__':
   model_params = {
          "activation": "relu",
          "l2_penalty": 1e-3,
          "dropout_prob": 0.3,
          "filter_factor": 4,
          "adam_lr": 1e-4,
          "n_classes": 1 }
   net = Unet(model_params=model_params)
   model = net.model()
   print(model.summary())
