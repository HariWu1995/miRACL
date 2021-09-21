import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import torch
import yaml
from difflib import SequenceMatcher
from string import punctuation

try:
    from nltk import sent_tokenize
except Exception:
    import nltk
    nltk.download('all')
    from nltk import sent_tokenize

logger = logging.getLogger(__name__)


def torch2np(tensor: torch.Tensor) -> np.array:
    if torch.cuda.is_available():
        tensor = tensor.cpu()
    return tensor.numpy()


def clean_punctuations(text: str) -> str:
    punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    text = re.sub(r"[%s]+" % punc+punctuation, "", text)
    return text


def normalize_punctuations(text: str) -> str:
    puncs_map = [
        (['＂', '〃', '｢', '｣', '『', '』', '〝', '〞', '“', '”', '‟'], '"'),
        (['＇', '｀', "‘", "’", '‛'], "'"),
        (['（', '｟', '《', '【', '〔', '〖', '〘', '〚'], '('),
        (['）', '｠', '》', '】', '〕', '〗', '〙', '〛'], ')'),
        ('［', '['), ('］', ']'), ('｛', '{'), ('｝', '}'), ('｜', '|'),
        ('：', ':'), ('；', ';'), ('？', '?'), ('！', '!'),
        (['～', '〜', '〰'], '~'),
        (['､', '〟', '，', '„'], ','),
        (['｡', '‧'], '.'), (['…', '﹏'], '...'),
        ('＃', '#'), ('＄', '$'), ('％', '%'), ('＆', '&'), ('＠', '@'),
        ('＜', '<'), ('＞', '>'), ('＝', '='), ('＿', '_'), ('＾', '^'),
        ('＊', '*'), ('＋', '+'), ('－', '-'), ('／', '/'), ('＼', "\\"),
    ]
    for ps_strange, p_norm in puncs_map:
        if isinstance(ps_strange, str):
            ps_strange = [ps_strange]
        for p_strange in ps_strange:
            text = text.replace(p_strange, p_norm)
    return text


#####################
# For CONFIGURATION #
#####################

def load_config(config_path: str):
    """
    load config file (yaml)

        Parameters
        ----------
            config_path: config path to yaml file

        Returns
        -------
        config : dict
    """

    logger.info(f" Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.full_load(f)

    return config


def setup_folders(config: dict):
    """
    ensure required folders in docker environment are created for
        training and predict (based on config file 'paths')

        Parameters
        ----------
        config : dict containing paths for creating logs, artefacts and outputs

        Returns
        -------
        log_path : pathlib Path
            folder for storing logs

        artefacts_path : pathlib Path
            folder for storing artefacts

        output_path : pathlib Path
            folder for storing outputs
    """
    from pathlib import Path

    for key, value in config['paths'].items():
        os.makedirs(value, exist_ok=True)

    log_path = Path(config['paths']['logs'])
    artefacts_path = Path(config['paths']['artefacts'])
    output_path = Path(config['paths']['output'])
    evaluate_path = Path(config['paths']['evaluate'])

    return log_path, artefacts_path, output_path, evaluate_path


def setup_root_logger(log_path:str, level:str = 'INFO'):
    """
    helper function to set up root logger with file output

    Parameters
    ----------
        log_path : folder where logs are saved
        level : logging level for root logger and both console and file handlers
    """

    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': level,
                'formatter': 'standard',
                'filename': log_path
            },
        },
        'root': {
            'level': level,
            'handlers': ['console', 'file']
        }
    }

    logging.config.dictConfig(config)


################
# For TRAINING #
################

def count_parameter(trainable_variables):
    total_parameters = np.sum(
        [np.prod(var.get_shape().as_list()) for var in trainable_variables]
    )
    return (total_parameters)


def min_max_normal(tensor):
    dim = tf.shape(tensor)[-1]
    max_value = tf.reduce_max(tensor, -1, keepdims=True)
    max_value = tf.tile(max_value, [1, 1, dim])
    min_value = tf.reduce_min(tensor, -1, keepdims=True)
    min_value = tf.tile(min_value, [1, 1, dim])
    norm_tensor = (tensor-min_value) / (max_value-min_value+1e-6)
    return norm_tensor


def z_score_normal(tensor):
    dim = tf.shape(tensor)[-1]
    axes = [2]
    mean, variance = tf.nn.moments(tensor, axes, keep_dims=True)
    std = tf.sqrt(variance)
    mean = tf.tile(mean, [1, 1, dim])
    std = tf.tile(std, [1, 1, dim])
    norm_tensor = (tensor - mean) / (std + 1e-6)
    return norm_tensor


def plot_history(history, figsize=(6, 9), return_figure: bool=True, **kwargs):
    """
    Plot the training history of one or more models.
        This creates a column of plots, with one plot for each metric recorded during training, 
        with the plot showing the metric vs. epoch. If multiple models have been trained (that is, 
        a list of histories is passed in), each metric plot includes multiple train and validation series.
        Validation data is optional (it is detected by metrics with names starting with `val_`).

    Args:
        history: the training history, as returned by :meth:`tf.keras.Model.fit`
        individual_figsize (tuple of numbers): the size of the plot for each metric
        return_figure (bool): if True, then the figure object with the plots is returned, None otherwise.
        kwargs: additional arguments to pass to :meth:`matplotlib.pyplot.subplots`

    Returns:
        :class:`matplotlib.figure.Figure`: The figure object with the plots if ``return_figure=True``, None otherwise

    Reference:
        https://github.com/stellargraph/stellargraph/blob/develop/stellargraph/utils/history.py
    """

    # explicit colours are needed if there's multiple train or multiple validation series, because
    # each train series should have the same color. This uses the global matplotlib defaults that
    # would be used for a single train and validation series.
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_train = colors[0]
    color_validation = colors[1]

    if not isinstance(history, list):
        history = [history]

    def remove_prefix(text, prefix):
        return text[text.startswith(prefix) and len(prefix) :]

    metrics = sorted({remove_prefix(m, "val_") for m in history[0].history.keys()})

    height, width = figsize
    overall_figsize = (width, len(metrics)*height)

    # plot each metric in a column, so that epochs are aligned (squeeze=False, so we don't have to
    # special case len(metrics) == 1 in the zip)
    fig, all_axes = plt.subplots(
        len(metrics), 1, squeeze=False, sharex="col", figsize=overall_figsize, **kwargs
    )

    has_validation = False
    for ax, m in zip(all_axes[:,0], metrics):
        for h in history:
            # summarize history for metric m
            ax.plot(h.history[m], c=color_train)

            try:
                val = h.history["val_" + m]
            except KeyError:
                # no validation data for this metric
                pass
            else:
                ax.plot(val, c=color_validation)
                has_validation = True

        ax.set_ylabel(m, fontsize="x-large")

    # don't be redundant: only include legend on the top plot
    labels = ["train"]
    if has_validation:
        labels.append("validation")
    all_axes[0, 0].legend(labels, loc="best", fontsize="x-large")

    # ... and only label "epoch" on the bottom
    all_axes[-1, 0].set_xlabel("epoch", fontsize="x-large")

    # minimise whitespace
    fig.tight_layout()

    if return_figure:
        return fig


def log_summary(model, line_length=None, positions=None):
    """
    Log a summary of a model.

    Args:
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[.33, .55, .67, 1.]`.

    Return: text of model summary
    """

    def log_row(fields, positions):
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i]-len(line))
        return line+'\n'

    def log_layer_summary(layer):
        """
        Log a summary for a single layer.
        Args:
            layer: target layer.
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        except RuntimeError:  # output_shape unknown in Eager mode.
            output_shape = '?'
        name = layer.name
        cls_name = layer.__class__.__name__
        if not layer.built and not getattr(layer, '_is_graph_network', False):
            # If a subclassed model has a layer that is not called in Model.call, the
            # layer will not be built and we cannot call layer.count_params().
            params = '0 (unused)'
        else:
            params = layer.count_params()
        fields = [name + ' (' + cls_name + ')', output_shape, params]
        return log_row(fields, positions)

    def log_layer_summary_with_connections(layer):
        """
        Log a summary for a single layer (including topological connections).
        Args:
            layer: target layer.
        """
        summary = ''
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = 'multiple'
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue

            for inbound_layer, node_index, tensor_index, _ in node.iterate_inbound():
                connections.append(f'{inbound_layer.name}[{node_index}][{tensor_index}]')

        name = layer.name
        cls_name = layer.__class__.__name__
        if not connections:
            first_connection = ''
        else:
            first_connection = connections[0]
        fields = [
            name + ' (' + cls_name + ')', output_shape, layer.count_params(), first_connection
        ]
        summary += log_row(fields, positions)
        if len(connections) > 1:
            for i in range(1, len(connections)):
                fields = ['', '', '', connections[i]]
                summary += log_row(fields, positions)
        return summary

    if model.__class__.__name__ == 'Sequential':
        sequential_like = True
    elif not model._is_graph_network:
        # We treat subclassed models as a simple sequence of layers, for logging purposes.
        sequential_like = True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or \
                (len(v) == 1 and len(tf.nest.flatten(v[0].keras_inputs)) > 1):
                # if the model has multiple nodes or if the nodes have multiple inbound_layers,
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        line_length = line_length or 65
        positions = positions or [.45, .85, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #']
    else:
        line_length = line_length or 98
        positions = positions or [.33, .55, .67, 1.]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    summary = f'Model: "{model.name}"\n'
    summary += '_' * line_length + '\n'
    summary += log_row(to_display, positions)
    summary += '=' * line_length + '\n'

    layers = model.layers
    for i in range(len(layers)):
        if sequential_like:
            summary += log_layer_summary(layers[i])
        else:
            summary += log_layer_summary_with_connections(layers[i])
        if i == len(layers) - 1:
            summary += '=' * line_length + '\n'
        else:
            summary += '_' * line_length + '\n'

    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)

    summary += 'Total params: {:,}\n'.format(trainable_count + non_trainable_count)
    summary += 'Trainable params: {:,}\n'.format(trainable_count)
    summary += 'Non-trainable params: {:,}\n'.format(non_trainable_count)
    summary += '_' * line_length + '\n'
    return summary


def count_params(weights):
    """
    Count the total number of scalars composing the weights.
    Args:
        weights: An iterable containing the weights on which to compute params
    Returns:
        The total number of scalars composing the weights
    """
    unique_weights = {id(w): w for w in weights}.values()
    weight_shapes = [w.shape.as_list() for w in unique_weights]
    standardized_weight_shapes = [
        [0 if w_i is None else w_i for w_i in w] for w in weight_shapes
    ]
    return int(sum(np.prod(p) for p in standardized_weight_shapes))


#################
# For INFERENCE #
#################

from src.sub_processes.convert_format import attribute_mapper, category_mapper

# Define useful variables
word_pattern = re.compile(r'\w+')
punct_pattern = re.compile(f"[{punctuation}]")
sentiment_dict = {
    0: 'positive',
    1: 'negative',
    2: 'neutral',
    # 4: 'conflict'
    # 0: 'background',
}
term_dict = {
    1: 'Beginning',
    2: 'Inside',
    0: 'Outside',
}
attribute_dict = {v: k for k,v in attribute_mapper.items()}
category_dict = {v: k for k,v in category_mapper.items()}
category_dict[0], attribute_dict[0] = 'Background', 'Background'


def split_words(text: str) -> list:
    # print(f"\n{text}")

    # Split sentence into phrases by punctuation
    punct_locs = [(p.start(), p.end()) for p in punct_pattern.finditer(text)]
    if len(punct_locs)==0:
        phrases_dict = {0: [0, len(text), text]}
    else:
        phrases_dict = dict()
        phrase_idx = 0
        last_punct_end = 0
        for p_i, punct_loc in enumerate(punct_locs):
            current_punct_start, current_punct_end = punct_loc
            if p_i == 0:
                if current_punct_start > 0:
                    phrases_dict[phrase_idx] = [0, current_punct_start, text[:current_punct_start]]
                    phrase_idx += 1
            elif p_i != 0:
                phrases_dict[phrase_idx] = [last_punct_end, current_punct_start, text[last_punct_end:current_punct_start]]
                phrase_idx += 1
            phrases_dict[phrase_idx] = [current_punct_start, current_punct_end, text[current_punct_start:current_punct_end]]
            phrase_idx += 1
            if p_i == len(punct_locs)-1:
                if current_punct_end < len(text):
                    phrases_dict[phrase_idx] = [current_punct_end, len(text)-1, text[current_punct_end:]]
            last_punct_end = current_punct_end

    # Split phrases into words (offset by sentence, not by current phrase)
    words_dict = dict()
    word_idx = 0
    for phrase_idx in range(len(phrases_dict)):
        phrase_start, phrase_end, phrase = phrases_dict[phrase_idx]
        if phrase_end-phrase_start == 1: # case of punctuation
            words_dict[word_idx] = phrases_dict[phrase_idx]
            word_idx += 1
        phrase_words_dict = {
            w_i+word_idx: [w.start()+phrase_start, w.end()+phrase_start, w.group(0)] \
                for w_i, w in enumerate(word_pattern.finditer(phrase))
        }
        word_idx += len(phrase_words_dict)
        words_dict.update(phrase_words_dict)

    return [word_data[2] for word_data in words_dict.values()]


def position_matrix(sen_len: int, max_len: int) -> np.array:
    A = np.zeros([max_len, max_len], dtype=np.float32)
    for i in range(sen_len):
        for j in range(sen_len):
            if i == j:
                A[i][j] = 0.
            else:
                distance = abs(i-j)
                A[i][j] = 1 / (np.log2(2+distance))
                # A[i][j] = 1 / (abs(i-j))
    return A


def read_data(sentences, opt, encoder):
    if not isinstance(sentences, (list, tuple)):
        sentences = [sentences]

    # Clean
    sentences = [normalize_punctuations(s) for s in sentences]

    # Encode
    tokenizer = encoder.tokenizer
    print("\n\nEncoding sentences ...")
    embeddings = encoder.encode(sentences, output_value='token_embeddings')
    embeddings_padded = np.zeros(shape=(len(embeddings), opt.max_sentence_len, opt.embedding_dim))
    for e_id, e in enumerate(embeddings):
        embeddings_padded[e_id, :e.shape[0]-2, :] = e[1:-1, :] # remove 2 tokens nonsense

    masks, position_matrices, tokens = [], [], []
    for sentence in sentences:

        tokens_ = tokenizer.tokenize(sentence)
        n_pads = opt.max_sentence_len - len(tokens_)

        masks += [[1.]*len(tokens_) + [0.]*n_pads]
        position_matrices += [position_matrix(len(tokens_), opt.max_sentence_len)]
        tokens += [tokens_[:opt.max_sentence_len]]
    return embeddings_padded, np.array(masks), np.array(position_matrices), tokens


# get html element
def color_str(s: str, color: str='black', join_token: str=' '):
    if s == ' ':
        return "<text style=color:#000000;padding-left:10px;background-color:{}> </text>".format(color)
    else:
        s = s[2:] if s.startswith("##") else join_token+s
        return "<text style=color:#000000;background-color:{}>{}</text>".format(color, s)
        # return "<u><font color={}><text style=color:#000>{} </text></u>".format(color, s)


def rgb2hex(r: int, g: int, b: int) -> str:
    r = max(min(r, 255), 0)
    g = max(min(g, 255), 0)
    b = max(min(b, 255), 0)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex2rgb(hex: str) -> list:
    return list(map(ord, hex[1:].decode('hex')))


# print html
def print_color(t: list or tuple) -> str:
    # from IPython.display import display, HTML as html_print
    # display(html_print(''.join([color_str(str(ti), color=ci) for ti, ci in t])))
    return ''.join([color_str(str(t_i), color=c_i) for t_i, c_i in t])


def scale_value(value: int, max_value: int=255, min_value: int=200, reverse_value: bool=True) -> int:
    if reverse_value:
        value = 1 - value
    value = max(min(value, 1.0), 0.0)
    value = value * (max_value-min_value) + min_value
    return int(value)


# get appropriate color for value
def get_color(value: float, label: str='background') -> str:

    # assign color code for background
    color_code = "#FFFFFF"

    # if not background, overwrite color code
    if label == 'opinion':
        value = scale_value(value, 200, 128)
        color_code = rgb2hex(value, value, value) # gray
    elif 'aspect' in label:
        value = scale_value(value)
        if 'pos' in label:
            color_code = rgb2hex(0, value, 0) # green
        elif 'neg' in label:
            color_code = rgb2hex(value, 0, 0) # red
        elif 'neu' in label:
            color_code = rgb2hex(value, value, 0) # yellow
    return color_code


def visualize(results_list: list or tuple) -> str:
    text_colors = []
    for t, v, l in results_list:
        # print(t, v, l)
        text_color = [t, get_color(v, l)]
        # print(f'\t{text_color[1]}')
        text_colors += [text_color]
    return print_color(text_colors)


def split_documents(documents: list or tuple):
    # Split documents into multiple single-sentences 
    # and record the reversed mapping
    sent2doc = dict()
    sentences = []
    for doc_i, doc in enumerate(documents):
        doc = normalize_punctuations(doc)
        sentences_in_doc = sent_tokenize(doc)
        sentences += sentences_in_doc
        for _ in sentences_in_doc:
            sent2doc[len(sent2doc)] = doc_i
    return sentences, sent2doc


def reverse_unk(tokens_list: list, sentences: list or str) -> list:

    if isinstance(sentences, str) and isinstance(tokens_list[0], str):
        sentences = [sentences]
        tokens_list = [tokens_list]

    tokens_reversed = []
    for sentence, tokens in zip(sentences, tokens_list):

        # Post-process tokens
        for t_i in range(len(tokens)):
            if '[UNK]' in tokens[t_i]:
                tokens[t_i] = f'UNK_{t_i}'
        sentence_reconstructed = ' '.join(tokens).replace(' ##', '')

        # Matching part-of-sentences
        matcher = SequenceMatcher(a=sentence, b=sentence_reconstructed)
        match_df = pd.DataFrame(columns=['%s_%s' % (b,a) for a in ['A', 'B'] for b in ['start', 'end']])
        for m_i, m_block in enumerate(matcher.get_matching_blocks()):
            match_df.loc[m_i] = [m_block[0], m_block[0]+m_block[2], m_block[1], m_block[1]+m_block[2]]

        # Reverse-mapping [UNK] token
        for m_i in range(len(match_df)):
            if m_i == 0:
                if match_df.loc[m_i, 'start_A'] == 0:
                    continue
                word_start, token_start = 0, 0
            else:
                word_start = match_df.loc[m_i-1, 'end_A']
                token_start = match_df.loc[m_i-1, 'end_B']
            word_end = match_df.loc[m_i, 'start_A']
            token_end = match_df.loc[m_i, 'start_B']

            unk_word = sentence[word_start:word_end]
            unk_token = sentence_reconstructed[token_start:token_end]
            if 'UNK' in unk_token:
                unk_pattern = re.compile(pattern="UNK_\d+")
                t_i = int(unk_pattern.search(unk_token).group()[4:])
                tokens[t_i] = unk_word

        tokens_reversed.append(tokens)
    return tokens_reversed


def decode_results(tokens_in_doc, sent2doc,
                   aspect_probs, opinion_probs, sentiment_probs) -> pd.DataFrame:
    results_df = pd.DataFrame(columns=['doc_id', 'token', 'aspect', 'aspect_prob', 'opinion', 'opinion_prob', 'sentiment', 'sentiment_prob'])
    r_id = 0
    for sentence_id, (tokens, aspect, opinion, sentiment) in \
        enumerate(zip(tokens_in_doc, aspect_probs, opinion_probs, sentiment_probs)):
        for t_idx, token in enumerate(tokens):
            results_df.loc[r_id] = [
                sent2doc[sentence_id], token,
                term_dict[np.argmax(aspect[t_idx])], np.max(aspect[t_idx]),
                term_dict[np.argmax(opinion[t_idx])], np.max(opinion[t_idx]),
                sentiment_dict[np.argmax(sentiment[t_idx])], np.max(sentiment[t_idx]),
            ]
            r_id += 1
    return results_df


def format_results(results_df) -> dict:

    doc_results = []
    for doc_id, doc_df in results_df.groupby(by=['doc_id']):
        results = doc_df.values.T.tolist()
        colors, labels = [], []

        _, tokens, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs = results
        for aspect, aspect_prob, opinion, opinion_prob, sentiment, sentiment_prob \
            in zip(aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs):
            if aspect != 'Outside':
                label = f'aspect_{sentiment}'
                colors.append(get_color(sentiment_prob*aspect_prob, label))
                labels.append(label)
            elif opinion != 'Outside':
                label = 'opinion'
                colors.append(get_color(opinion_prob, label))
                labels.append(label)
            else:
                colors.append("#FFFFFF")
                labels.append('background')

        doc_results.append({
            'id': doc_id,
            'tokens': tokens,
            'colors': colors,
            'labels': labels,
        })

    return {'aspect-based sentiment': doc_results}


def write_log(results_df, fn: str, threshold: float=0.5):
    logger = open(fn, 'w')
    for doc_id, doc_df in results_df.groupby(by=['doc_id']):
        # print(doc_df)
        # doc_df.reset_index(inplace=True)
        doc_results = doc_df.values.T.tolist()
        _, words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs = doc_results
        logger.write(f'\n\n\n{"-"*29}\nSentence {doc_id+1:02d}: {len(words)} words - {" ".join(words)}\n')
        for w_idx, (word, aspect, aspect_prob, opinion, opinion_prob, sentiment, sentiment_prob) \
            in enumerate(zip(words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs)):
            logger.write(f'\tWord {w_idx+1:02d}: {word}\n')
            logger.write(f'\t\t{"Aspect":>10s}: {aspect if aspect_prob > threshold else None} - {max(aspect_prob, threshold):.3f}\n')
            logger.write(f'\t\t{"Opinion":>10s}: {opinion if opinion_prob > threshold else None} - {max(opinion_prob, threshold):.3f}\n')
            logger.write(f'\t\t{"Sentiment":>10s}: {sentiment if sentiment_prob > threshold else None} - {max(sentiment_prob, threshold):.3f}\n')
    logger.close()


def write_html(results_df: pd.DataFrame, fn: str):
    n_docs = len(results_df.doc_id.unique())
    logger = open(fn, 'w', encoding='utf-8')
    for doc_id, doc_df in results_df.groupby(by=['doc_id']):
        text_head = f"{'<br>'*5}[{doc_id+1:03d}/{n_docs:03d}]{'<br>'*1}"
        doc_results = doc_df.values.T.tolist()

        results = [] # list of tuples <word, confidence_score, label>
        _, words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs = doc_results
        for word, aspect, aspect_prob, opinion, opinion_prob, sentiment, sentiment_prob \
            in zip(words, aspects, aspect_probs, opinions, opinion_probs, sentiments, sentiment_probs):
            result = [word]
            if aspect != 'Outside':
                result.extend([sentiment_prob*aspect_prob, f'aspect_{sentiment}'])
            elif opinion != 'Outside':
                result.extend([opinion_prob, 'opinion'])
            else:
                result.extend([0.0, 'background'])

            # print(result)
            results.append(result)

        text_body = visualize(results)
        logger.write(text_head+text_body)
    logger.close()
    return True


def dict2html(doc_results: dict, fn: str):
    import io

    doc_results = doc_results['aspect-based sentiment']
    logger = io.open(fn, 'w', encoding='utf-8')
    n_docs = len(doc_results)
    for doc_id, doc_result in enumerate(doc_results):
        results_list = list(zip(doc_result['tokens'], doc_result['colors']))
        text_head = f"{'<br>'*5}[{doc_id+1:02d}/{n_docs:02d}]{'<br>'*1}"
        text_body = ''.join([color_str(str(t_i), color=c_i) for t_i, c_i in results_list])
        logger.write(text_head+text_body)
    logger.close()


