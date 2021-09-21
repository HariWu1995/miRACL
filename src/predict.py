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


# Samples for prediction
documents = [
    # 'dessert was also to die for',
    # 'sushi so fresh that it crunches in your mouth',
    # 'in fact , this was not a nicoise salad and was barely eatable',
    # "the two waitress 's looked like they had been sucking on lemons",
    "the absence of halal food - not even for room service",
    "la foresto de halalaj manĝaĵoj - eĉ ne por ĉambroservo",
    "عدم وجود الطعام الحلال - ولا حتى لخدمة الغرف",
    "អវត្ដមាននៃអាហារហាឡាល់ - មិនសូម្បីតែសម្រាប់សេវាកម្មបន្ទប់",
    "ການຂາດອາຫານຮາລານ - ບໍ່ແມ່ນແຕ່ ສຳ ລັບການບໍລິການຫ້ອງ",
    "халал тағамның болмауы - тіпті бөлме қызметтері үшін де емес",
    "отсутствие халяльной еды - даже для обслуживания номеров",
    "die afwesigheid van halal-kos - nie eens vir kamerdiens nie",
    "l'assenza di cibo halal - nemmeno per il servizio in camera",
    "ハラルフードがない-ルームサービスでもない",
    "할랄 음식의 부재-룸 서비스조차도",
    "la ausencia de comida halal, ni siquiera para el servicio de habitaciones",
    "sự vắng mặt của thức ăn halal - thậm chí không có dịch vụ ăn uống tại phòng",
    # "Have to travel out in order to get food",
    # "Smell of the pillows... smelt like someone odour",
    # " Very noisy outside the room, found a cockroaches in bathroom, the condition did not works whole nights, very hot can't sleep",
    # "I had to stay here due to holiday inn transferring me here because they were closed for renovations. First I am pist because this hotel stinks of weed, my room was not very clean and due to Covid you would think the room would be super clean but nope wrappers all over the place towels had stains, to top it off I even found bugs in my room. I am disgusted. The service is horrible. “There was never a manager on duty” I even reached out to them in email and still no reply from them so they clearly don’t care. Avoid this hotel there are so many other options by the airport that this one poor excuse for cleanliness and bugs they do not deserve a dime. They don’t fix their problems and a manager is never reachable",
    # "First impression is the hotel seem to be in need of an upgrade. The grounds did not feel welcoming on the exterior. The interior had carpet coming up in the hallway, I was on the third floor. It had a bad smell that hits you in the face as soon as you get off the elevator. The rooms was decent with a nice size television, desk and a refrigerator but lacked cleanliness. We couldn't shower because the tubes were GROSS. It looked as if it hadn't been properly cleaned for months! You can see the filth buildup YUCK! This is very concerning considering the month I traveled was during the covid-19 pandemic. If this hotel is not properly cleaning guest rooms than are they really practicing safe measures during a global coronavirus pandemic?",
    # "Small rooms, restaurant offers the best of microwaved food and wifi is poor. Staff set engaged, but this establishment needs investment and attention to the the customer experience. Plenty of examples where the site could use a goos cleaning - including the restaurant.",
    # "I had a horrible check-in experience at this crown plaza. The manager at night shift was exceptionally rude. Just because it was night and I was tired, I stayed there. I checked out next day and went to The Renaissance across the street.",
    # "DIRTY FILTHY DISGUSTING!!! Hair and mold in the bathroom, DIRTY carpeting, smells of cigarette smoke and my daughter woke up with bug bites all over her legs!!! Front desk was an absolute joke! Unprofessional rude and lazy!! Travelers BEWARE!!",
    # "Called to say my flight is cancelled because of weather ,can you change to next day or refund.before I could complete the sentence they cancelled my reservation and hung up.i know the hotel room was given to somebody else.i cannot believe the service was from very reputable company like yours",
    # "The value for the room and the service was very good but the Furnishings in the room is very outdated and more out. The carpet has been replaced and the linen and the bathtub was spotless. Restaurant bar",
    # "The Crowne Plaza is located near the newark airport. The hotel offers a transfer ( i got it on my way back). The rooms are small but the bed is very comfortable. Bathroom regular. Also offers a transfer to the outlet nearby but only in 2 specific times a day.",
    # "We stayed one night (thankfully) as there was a lot of noise from airplanes taking off and landing and from traffic on the road nearby. The room was very nice with comfortable bed. The shower was over the bath",
    # "I visited this hotel with 6 family members in jan 2020. we reached jetlagged early in the morning to be greeted by an extremely rude lady whose name started with Q. I saw her even mocking a few clients. Rooms were clean. Sleep quality was nice Not many eating options around hotel for breakfast, except the hotel itself. In evening one can walk out towards quay and be delighted with so many restaurants. over all a an average hotel BUT the RUDEST STAFF i have ever seen. STAY AWAY IF YOU ANYOTHER OPTION.",
    # "Hotel was very crowded and so called club lounge was so crowded that we couldn't use 20 minute wait for breakfast in main restaurant Hotel room small and basic - not luxury Pool good and hotel location excellent",
    # "The hotel is actually Robertson Quay not Clarke Quay as the name claims. I had booked a room with a king size bed but they could only give me twin beds on the first night so I had to move rooms on the second day. All of the rooms I saw were tired with very bland decor and badly in need of a refresh. I also experienced a lot of noise from neighbouring rooms",
    # "I do no understand why you are charging me USD 100 (66% of original room charge) because I have Netherlands nationality but booked my room stating my residential address in Thailand, where I have lived for the last 13 years",
    # "Check in was appalling ! Checked into a deluxe room but was given two single beds!! Went downstairs to speak to reception and they told me only room they have is a smoking room which was not practical!!! Then had to sleep there and next day await a room change!!! Which was chased by us as no one remembered the next day!!",
    # "I would not recommend this hotel, it is seriously understaffed the restaurant is small for the size of the hotel which results in the tables being too close together. The restaurant staff tried their best but there just weren't enough of them",
    # "nice bar and front desk staff members happy faces they made me feel like a vip. update! hotel is dark and old. bathroom was tiny, dark and poor design. elevator was slow. hotel facilities and staff were excellent",
]


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
    output_file = os.path.join(predicts_dir, f'case_study_{opt.task}')
    print(f'\n\nWriting result to \n\t{output_file}.json\n\t{output_file}.html ...')
    doc_results = format_results(results_df)
    with open(output_file+'.json', 'w') as f_writer:
        json.dump(doc_results, f_writer, indent=4)
    dict2html(doc_results, output_file+'.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Prediction')
    parser.add_argument('-c', '--config-path', default='production/model_config.yml', type=str, help='Config path')

    args, unk_args = parser.parse_known_args()
    predict(parser, args)


    ##########################################
    #   Executive Time on Local Machine:     #
    #       Tokenize 13 samples in 0.22s     #
    #        Predict 13 samples in 2.27s     #
    ##########################################

