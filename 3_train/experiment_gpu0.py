import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

module_root = '..'
sys.path.append(module_root)

from model.train import Experiments
from settings import intermediate_folder

JSON_CONFIG_FILE = os.path.abspath(os.path.join('.', 'configs', 'experiment5.json'))

if __name__ == '__main__':
    experiments = Experiments(JSON_CONFIG_FILE, intermediate_folder)
    experiments.run()
