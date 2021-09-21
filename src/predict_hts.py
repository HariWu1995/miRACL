import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import time
import argparse
from pathlib import Path

import random
import numpy as np
import tensorflow as tf
tf.autograph.set_verbosity(3) # 0: debug, 1: info, 2: warning, 3: error

from src.models.encoder import Encoder
from src.models.RACL import RACL
from src.utils import (
    load_config,
    split_documents, read_data, reverse_unk,
    decode_results, format_results, dict2html
)


def load_basic_arguments(parser):
    # Define arguments
    parser.add_argument('--model', default='racl', type=str, help='model name')
    parser.add_argument('--task', default='hotel', type=str, help='task name')
    parser.add_argument('--max_sentence_len', default=156, type=int, help='maximum number of words in sentence')
    parser.add_argument('--embedding_dim', default=768, type=int, help='embedding dimension')
    parser.add_argument('--n_interactions', default=6, type=int, help='number of RACL blocks to interact')
    parser.add_argument('--n_filters', default=96, type=int, help='number of filters in convolution')
    parser.add_argument('--kernel_size', default=11, type=int, help='kernel size in convolution')
    parser.add_argument('--random_seed', default=4_10_20, type=int, help='random seed')
    parser.add_argument('--include_opinion', default=True, type=bool, help='whether to use opinion for model')
    parser.add_argument('--random_type', default='normal', type=str, help='random type: uniform or normal (default)')
    parser.add_argument('--ckpt', default=798, type=int, help='checkpoint id to load weights')
    opt = parser.parse_args()

    opt.n_classes = 3
    opt.is_training = False
    opt.is_evaluating = False
    opt.label_smoothing = False
    opt.keep_prob_1, opt.keep_prob_2 = 1., 1.

    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    tf.random.set_seed(opt.random_seed)
    return opt


def predict(parser, args):
    """
    Predict from command line and return response output as html + json

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

    # Define useful directories
    predicts_dir = config["paths"]["predictions"]
    artefacts_dir = config["paths"]["artefacts"]
    checkpoint_dir = config["paths"]["checkpoint"]
    opt.ckpt_path = os.path.join(checkpoint_dir, f"RACL-epoch={opt.ckpt:03d}.h5")

    # Split document into sentences
    sentences, sent2doc = split_documents(documents)
    opt.batch_size = len(sentences)

    # Load Tokenizer and Encoder
    print(f"\n\n\nLoading Encoder ...")
    sbert_version = 'distilUSE'
    sbert_dir = os.path.join(artefacts_dir, sbert_version)
    print(sbert_dir)
    encoder = Encoder(sbert_dir)

    # Tokenize
    start_time = time.time()
    embeddings, sentences_mask, position_matrices, tokens_in_doc = read_data(sentences, opt, encoder)
    embeddings = np.reshape(embeddings, (opt.batch_size, opt.max_sentence_len, opt.embedding_dim))
    tokens_in_doc = reverse_unk(tokens_in_doc, sentences)
    end_time = time.time()
    time_running = end_time - start_time
    run_time = f'\n\n\nTokenize {len(sentences)} samples in {time_running:.2f}s'
    print(run_time)

    # Load model
    model = RACL(opt)
    model.load_weights(opt.ckpt_path)

    # Predict
    start_time = time.time()
    aspect_probs, opinion_probs, sentiment_probs = model.predict(
        sentence=embeddings,
        word_mask=sentences_mask.reshape((opt.batch_size, opt.max_sentence_len)),
        position_att=position_matrices.reshape((opt.batch_size, opt.max_sentence_len, opt.max_sentence_len))
    )
    end_time = time.time()
    time_running = end_time - start_time
    run_time = f'\n\n\nPredict {len(sentences)} samples in {time_running:.2f}s'
    print(run_time)

    # Feed results into DataFrame
    results_df = decode_results(tokens_in_doc, sent2doc,
                                aspect_probs, opinion_probs, sentiment_probs)

    # Write logs
    output_file = os.path.join(predicts_dir, f'predict_{opt.language}')
    print(f'\n\nWriting result to \n\t{output_file}.json\n\t{output_file}.html ...')
    doc_results = format_results(results_df)
    with open(output_file+'.json', 'w') as f_writer:
        json.dump(doc_results, f_writer, indent=4)
    dict2html(doc_results, output_file+'.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Prediction')
    parser.add_argument('-c', '--config-path', default='production/model_config.yml', type=str, help='Config path')
    parser.add_argument('-l', '--language', default='english', type=str, help='Language to load file of comments')

    args, unk_args = parser.parse_known_args()

    with open(f'src/comments_{args.language}.txt', 'r', encoding="utf-8") as f:
        documents = f.read().splitlines()
        # print(documents)
    predict(parser, args)

