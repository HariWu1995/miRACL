import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import argparse
from pathlib import Path

import random
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(3) # 0: debug, 1: info, 2: warning, 3: error

from src.utils import load_config
from src.models.miRACL import MIRACL
from src.sub_processes.preprocess import MAX_SEQ_LEN
from src.sub_processes.convert_format import attribute_mapper, category_mapper


def load_basic_arguments(parser):
    # Define arguments
    parser.add_argument('--model', default='miRACL', type=str, help='model name')
    parser.add_argument('--n_loops', default=2, type=int, help='number of loops to repeat `n_epochs` for training procedure')
    parser.add_argument('--n_epochs', default=1, type=int, help='number of epochs for training per phase')
    parser.add_argument('--batch_size', default=8, type=int, help='number of samples per batch')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--n_interactions', default=4, type=int, help='number of RACL blocks to interact')
    parser.add_argument('--keep_prob_1', default=.89, type=float, help='keep prob for inputs')
    parser.add_argument('--keep_prob_2', default=.89, type=float, help='keep prob for tasks')
    parser.add_argument('--kernel_size', default=7, type=int, help='kernel size in convolution')
    parser.add_argument('--n_filters', default=32, type=int, help='number of filters in convolution')
    parser.add_argument('--optimizer', default='nadam', type=str, help='optimizer for model | default: SGD (Stochastic Gradient Descent)')
    parser.add_argument('--random_type', default='normal', type=str, help='random type: uniform or normal (default)')
    parser.add_argument('--random_seed', default=4_10_20, type=int, help='random seed')
    parser.add_argument('--ae_weight', default=11., type=float, help='weight of aspect loss')
    parser.add_argument('--oe_weight', default=13., type=float, help='weight of opinion loss')
    parser.add_argument('--sc_weight', default=1., type=float, help='weight of sentiment loss')
    parser.add_argument('--ac_weight', default=1., type=float, help='weight of attribute loss')
    parser.add_argument('--cc_weight', default=1., type=float, help='weight of category loss')
    parser.add_argument('--reg_weight', default=1e-4, type=float, help='weight of regularization loss')
    parser.add_argument('--include_opinion', default=True, type=bool, help='whether to use opinion for model')
    parser.add_argument('--label_smoothing', default=False, type=bool, help='label smoothing for regularization')
    parser.add_argument('--load_pretrained', default=False, type=bool, help='whether to load an existing checkpoint')
    parser.add_argument('-ckpt', '--load_ckpt', default=0, type=int, help='checkpoint id to load weights (only available if `load_pretrained` is True)')
    
    opt, _ = parser.parse_known_args()
    opt.max_sentence_len = MAX_SEQ_LEN
    opt.embedding_dim = 768
    opt.n_sentiments = 3
    opt.n_categories = len(category_mapper.keys()) + 1
    opt.n_attributes = len(attribute_mapper.keys()) + 1
    
    opt.is_training = True
    opt.is_evaluating = False

    opt.polarity_weights = [.4, .4, .2] # Positive-Negative-Neutral
    opt.term_weights = [.2, .5, .3] # Outside-Beginning-Inside
    category_weights = np.array([1e-3]+[1.]*(opt.n_categories-1)) / opt.n_categories # Background + len(category_mapper)
    attribute_weights = np.array([1e-3]+[1.]*(opt.n_attributes-1)) / opt.n_attributes # Background + len(attribute_mapper)
    opt.category_weights = list(category_weights.astype(np.float32))
    opt.attribute_weights = list(attribute_weights.astype(np.float32))

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    tf.random.set_seed(opt.random_seed)
    return opt


def train(parser, args):

    """
    Training pipeline and return saved model as h5
    ________________________________________________________________________________
    Parameters
        args :
            args.config_path : str
                path to config yml e.g. /production/model_config.yml
            args.log_level: str
                'debug', 'info', or 'warning' level for root logger and all handlers
    """
    config = load_config(Path(args.config_path))
    opt = load_basic_arguments(parser)

    for key, value in config["model_params"].items():
        print(f"Key: {key} - Value: {value}")
        opt.key = value

    # Assign pre-defined paths
    opt.logs_path = config["paths"]["logs"]
    opt.ckpt_path = config["paths"]["checkpoint"]
    opt.output_path = config["paths"]["output"]
    opt.evaluate_path = config["paths"]["evaluate"]
    opt.prediction_path = config["paths"]["predictions"]
    opt.train_path = os.path.join(config["paths"]["dataset"], 'train')
    opt.test_path = os.path.join(config["paths"]["dataset"], 'test')
    opt.val_path = os.path.join(config["paths"]["dataset"], 'val')

    for path in [opt.logs_path, opt.output_path, opt.ckpt_path, opt.evaluate_path, opt.prediction_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # Train
    start_time = time.time()
    model = MIRACL(opt)
    if opt.load_pretrained:
        ckpt_path = os.path.join(opt.ckpt_path, f"RACL-epoch={opt.ckpt:03d}.h5")
        if os.path.isfile(ckpt_path):
            model.logger.info(f'\n\n\nLoading weights @ {ckpt_path}')
            model.load_weights(ckpt_path)
    model.train()
    end_time = time.time()
    time_running = end_time - start_time
    run_hours = int(time_running//3600)
    run_minutes = int((time_running-run_hours*3600)//60)
    run_seconds = int(time_running - run_hours*3600 - run_minutes*60)
    run_time = f'\n\n\nTraining in {run_hours}h {run_minutes}m {run_seconds}s'
    model.logger.info(run_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('-c', '--config-path', default='production/model_config.yml', type=str, help='Config path')

    args, unk_args = parser.parse_known_args()
    train(parser, args)


    #############################################
    #   Executive Time on Kaggle GPU            #
    #       Epoch 001 - 140:   6h 14m 38s       #
    # ----------------------------------------- #
    #            Total Time:   6h 14m 38s       #
    #          Average Time:         161s       #
    # ----------------------------------------- #
    # 	Best Result @ epoch = 757               #
    #		    opinion_f1 = 0.9230769          #
	#	         aspect_f1 = 0.9233954          #
	#	      sentiment_f1 = 0.9176369          #
	#	           ABSA_f1 = 0.8612836          #
	#	       category_f1 = 0.8612836          #
	#         category_acc = 0.8612836          #
	#	      attribute_f1 = 0.8612836          #
	#        attribute_acc = 0.8612836          #
	#	         miRACL_f1 = 0.8612836          #
    #############################################



