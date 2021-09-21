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

from src.models.RACL import RACL
from src.utils import load_config


def load_basic_arguments(parser):
    # Define arguments
    parser.add_argument('--model', default='racl', type=str, help='model name')
    parser.add_argument('--max_sentence_len', default=156, type=int, help='maximum number of words in sentence')
    parser.add_argument('--embedding_dim', default=768, type=int, help='embedding dimension')
    parser.add_argument('--n_interactions', default=6, type=int, help='number of RACL blocks to interact')
    parser.add_argument('--keep_prob_1', default=.89, type=float, help='keep prob for inputs')
    parser.add_argument('--keep_prob_2', default=.89, type=float, help='keep prob for tasks')
    parser.add_argument('--n_filters', default=96, type=int, help='number of filters in convolution')
    parser.add_argument('--kernel_size', default=11, type=int, help='kernel size in convolution')
    parser.add_argument('--random_type', default='normal', type=str, help='random type: uniform or normal (default)')
    parser.add_argument('--random_seed', default=4_10_20, type=int, help='random seed')
    parser.add_argument('--label_smoothing', default=False, type=bool, help='label smoothing for regularization')
    parser.add_argument('--load_pretrained', default=False, type=bool, help='whether to load an existing checkpoint')
    parser.add_argument('--include_opinion', default=True, type=bool, help='whether to use opinion for model')
    parser.add_argument('--ckpt', default=798, type=int, help='checkpoint id to load weights')
    opt = parser.parse_args()

    opt.n_classes = 3
    opt.is_training = False
    opt.is_evaluating = True

    opt.term_weights = [.2, .5, .3] # Outside-Beginning-Inside
    opt.polarity_weights = [.3, .3, .4] # Positive-Negative-Neutral

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    tf.random.set_seed(opt.random_seed)
    return opt


def evaluate(parser, args):

    """Training pipeline and return saved model as h5

        Parameters
        ----------
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

    # Load model
    model = RACL(opt)
    model.load_weights(os.path.join(opt.ckpt_path, f"RACL-epoch={opt.ckpt:03d}.h5"))

    # Evaluate
    start_time = time.time()
    scores = model.evaluate()
    end_time = time.time()
    time_running = end_time - start_time
    run_time = f'\n\n\nEvaluation test-set in {time_running:.2f}s'
    print(run_time)
    print(f'\n\nEpoch {opt.ckpt}\n\topinion_f1={scores[0]:.7f}\n\taspect_f1={scores[1]:.7f}\n\tsentiment_acc={scores[2]:.7f}\n\tsentiment_f1={scores[3]:.7f}\n\tABSA_f1={scores[4]:.7f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('-c', '--config-path', default='production/model_config.yml', type=str, help='Config path')

    args, unk_args = parser.parse_known_args()
    evaluate(parser, args)


    #########################################
    #   Executive Time on Local Machine:    #
    #       Evaluation test-set in 35.09s   #
    #       Epoch 798                       #
    #               opinion_f1=0.9486386    #
    #                aspect_f1=0.9477455    #
    #            sentiment_acc=0.9632968    #
    #             sentiment_f1=0.9528047    #
    #                  ABSA_f1=0.9119170    #
    #########################################


