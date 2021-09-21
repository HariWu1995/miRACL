import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric

from sklearn.metrics import f1_score, accuracy_score


epsilon = K.epsilon()
term_code = {'begin': 1, 'inside': 2, 'outside': 0, }
idx2polarity = {0: 'background', 1: 'positive', 2: 'negative', 3: 'neutral', 4: 'conflict', }
polarity2idx = {v: k for k,v in idx2polarity.items()}


def mask_absa(y_aspect, y_sentiment, mask):
    # Mask background words and conflict-sentiment words
    #       for not to count in evaluation
    Ys_aspect, Ys_sentiment = [], []
    for seq_aspect, seq_sentiment, seq_mask in zip(y_aspect, y_sentiment, mask):
        labels_aspect, labels_sentiment = [], []
        for l_a, l_s, m in zip(seq_aspect, seq_sentiment, seq_mask):
            if m == 0:
                break
            labels_aspect.append(np.argmax(l_a))
            if not np.any(l_s):
                # all 0s means background or conflict-sentiment word
                #       -> be not counted for evaluation
                labels_sentiment.append(0)
            else:
                labels_sentiment.append(np.argmax(l_s)+1)

        Ys_aspect.append(labels_aspect)
        Ys_sentiment.append(labels_sentiment)
    return Ys_aspect, Ys_sentiment


def score_absa_single_sample(t_true, t_pred, s_true=[], s_pred=[], counters: dict={}, term_only: bool=False):
    n_relevants, n_corrects, n_predicteds = 0, 0, 0
    n_words = len(t_true)
    for j in range(n_words):
        if t_true[j] == term_code['begin']:
            n_relevants += 1
            if not term_only:
                if s_true[j] != polarity2idx['background']:
                    counters['overall'][idx2polarity[s_true[j]]] += 1

            if t_pred[j] == term_code['begin']:
                matching = True
                for k in range(j+1, len(t_true)):
                    if t_true[k] == term_code['inside'] and t_pred[k] == term_code['inside']:
                        continue
                    elif t_true[k] != term_code['inside'] and t_pred[k] != term_code['inside']:
                        break
                    else:
                        matching = False
                        break

                if matching:
                    n_corrects += 1
                    if not term_only:
                        if s_true[j] != polarity2idx['background']:
                            counters['gold'][idx2polarity[s_true[j]]] += 1
                            counters['pred'][idx2polarity[s_pred[j]]] += 1
                            if s_true[j] == s_pred[j]:
                                counters['correct'][idx2polarity[s_pred[j]]] += 1
                        else:
                            counters['pred']['conflict'] += 1

    for t_p in t_pred:
        if t_p == term_code['begin']:
            n_predicteds += 1

    if term_only:
        return n_relevants, n_corrects, n_predicteds
    return [n_relevants, n_corrects, n_predicteds], counters


def score_absa(terms_true, terms_pred,
               sentiments_true: list=[], sentiments_pred: list=[],
               average_method: str='macro', term_only: bool=False):
    # Define useful variables
    if not term_only:
        # Sentiment Distribution for Aspect / Opinion Terms:
        #          pred_count: predicted results that are correctly extracted
        #          gold_count: gold results that are correctly extracted
        #       correct_count: results that get both span & prediction correctly
        #       overall_count: ground-truth
        counters = {
            'gold': {'positive': 0, 'negative': 0, 'neutral': 0, },
            'pred': {'positive': 0, 'negative': 0, 'neutral': 0, 'conflict': 0},
            'correct': {'positive': 0, 'negative': 0, 'neutral': 0, },
            'overall': {'positive': 0, 'negative': 0, 'neutral': 0, }
        }

    # Do statistics
    n_corrects, n_predicteds, n_relevants = 0, 0, 0
    n_samples = len(terms_true)
    for i in range(n_samples):
        t_true, t_pred = terms_true[i], terms_pred[i]

        if term_only:
            sample_relevants, sample_corrects, sample_predicteds = score_absa_single_sample(t_true, t_pred, term_only=term_only)
        else:
            s_true, s_pred = sentiments_true[i], sentiments_pred[i]
            [sample_relevants, sample_corrects, sample_predicteds], \
                counters = score_absa_single_sample(t_true, t_pred, s_true, s_pred, counters, term_only)

        n_corrects += sample_corrects
        n_relevants += sample_relevants
        n_predicteds += sample_predicteds

    # Calculate evaluation metrics for Term (of Aspect or Opinion)
    term_P = n_corrects / (n_predicteds+epsilon) # precision
    term_R = n_corrects / (n_relevants+epsilon) # recall
    term_F1 = 2*term_P*term_R / (term_P+term_R+epsilon)

    if term_only:
        return term_F1

    sentiment_Acc, sentiment_F1, absa_F1 = score_sentiment_and_overall(n_predicteds, counters, average_method)
    return term_F1, sentiment_Acc, sentiment_F1, absa_F1


def score_sentiment_and_overall(n_predicteds: int, counters: dict, average_method: str='micro'):
    # Precision and Recall per each sentiment polarity
    positive_P = counters['correct']['positive'] / (counters['pred']['positive']+epsilon)
    positive_R = counters['correct']['positive'] / (counters['gold']['positive']+epsilon)

    negative_P = counters['correct']['negative'] / (counters['pred']['negative']+epsilon)
    negative_R = counters['correct']['negative'] / (counters['gold']['negative']+epsilon)

    neutral_P = counters['correct']['neutral'] / (counters['pred']['neutral']+epsilon)
    neutral_R = counters['correct']['neutral'] / (counters['gold']['neutral']+epsilon)

    # Calculate evaluation metrics for Sentiment
    n_corrects_sentiment = counters['correct']['positive'] + counters['correct']['negative'] + counters['correct']['neutral']
    n_corrects_aspect = counters['gold']['positive'] + counters['gold']['negative'] + counters['gold']['neutral']
    n_overall = counters['overall']['positive'] + counters['overall']['negative'] + counters['overall']['neutral']

    sentiment_Acc = n_corrects_sentiment / (n_corrects_aspect+epsilon)
    if average_method == 'micro':
        sentiment_P = (positive_P+negative_P+neutral_P) / 3.0
        sentiment_R = (positive_R+negative_R+neutral_R) / 3.0
        sentiment_F1 = 2*sentiment_P*sentiment_R / (sentiment_P+sentiment_R+epsilon)
    elif average_method == 'macro':
        positive_F1 = 2*positive_P*positive_R / (positive_P+positive_R+epsilon)
        negative_F1 = 2*negative_P*negative_R / (negative_P+negative_R+epsilon)
        neutral_F1 = 2*neutral_P*neutral_R / (neutral_P+neutral_R+epsilon)
        sentiment_F1 = (positive_F1+negative_F1+neutral_F1) / 3.0
    else:
        raise ValueError('average_method must be either micro or macro')

    # Calculate evaluation metrics for ABSA
    absa_P = n_corrects_sentiment / (n_predicteds-counters['pred']['conflict']+epsilon)
    absa_R = n_corrects_sentiment / (n_overall+epsilon)
    absa_F1 = 2*absa_P*absa_R / (absa_P+absa_R+epsilon)

    return sentiment_Acc, sentiment_F1, absa_F1


def evaluate_absa(aspects_true, aspects_pred,
                  opinions_true, opinions_pred,
                  sentiments_true, sentiments_pred,
                  mask, 
                  include_opinion: bool=True):
    aspects_true, sentiments_true = mask_absa(aspects_true, sentiments_true, mask)
    aspects_pred, sentiments_pred = mask_absa(aspects_pred, sentiments_pred, mask)
    absa_scores = score_absa(aspects_true, aspects_pred, sentiments_true, sentiments_pred)
    # aspect_f1, sentiment_acc, sentiment_f1, absa_f1 = absa_scores

    if include_opinion:
        opinions_true, _ = mask_absa(opinions_true, sentiments_true, mask)
        opinions_pred, _ = mask_absa(opinions_pred, sentiments_pred, mask)
        opinion_f1 = score_absa(opinions_true, opinions_pred, term_only=True)
        absa_scores = [opinion_f1] + list(absa_scores)

    return absa_scores


def evaluate_multilists(arrays_true, arrays_pred, mask, average_method: str='macro'):
    if isinstance(arrays_true, np.ndarray):
        arrays_true = [arrays_true]
        arrays_pred = [arrays_pred]
    elif not isinstance(arrays_true, (list, tuple)):
        raise ValueError(f"inputs must be np.ndarray / list / tuple, not {arrays_true.__class__}")

    average_method = average_method.lower()
    if average_method not in ['micro', 'macro']:
        print(f'average_method={average_method} is out-of-option, so average_method is set as micro')
        average_method = 'macro'

    def evaluate(list_true, list_pred, mask):
        # Remove background words for not to count in evaluation
        list_true, list_true_unmask = np.argmax(list_true, axis=-1), []
        list_pred, list_pred_unmask = np.argmax(list_pred, axis=-1), []
        for seq_true, seq_pred, seq_mask in zip(list_true, list_pred, mask):
            seq_true_unmask, seq_pred_unmask = [] , []
            for token_true, token_pred, token_mask in zip(seq_true, seq_pred, seq_mask):
                if token_mask == 0:
                    break
                if token_true == 0:
                    continue    # skip background
                seq_true_unmask.append(token_true)
                seq_pred_unmask.append(token_pred)

            list_true_unmask.extend(seq_true_unmask)
            list_pred_unmask.extend(seq_pred_unmask)

        list_score = [f1_score(list_true_unmask, list_pred_unmask, zero_division='warn', average=average_method),
                accuracy_score(list_true_unmask, list_pred_unmask)]
        return list_score

    scores = [evaluate(list_true, list_pred, mask) 
                   for list_true, list_pred in zip(arrays_true, arrays_pred)]
    return sum(scores, []) # flatten nested lists


class F_score(Metric):

    def __init__(self, beta: int=1, threshold: float=0.5, **kwargs):
        super().__init__(**kwargs) # handle base args (e.g. dtype)
        self.beta = int(beta)
        self.threshold = float(threshold)
        self.count_positives = create_positive_counters(beta, threshold)
        self.n_true_positives = self.add_weight("n_true_positives", initializer="zeros")
        self.n_possible_positives = self.add_weight('n_possible_positives', initializer='zeros')
        self.n_predicted_positives = self.add_weight('n_predicted_positives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_positives, predicted_positives, possible_positives = self.count_positives(y_true, y_pred)
        self.n_true_positives.assign_add(true_positives)
        self.n_possible_positives.assign_add(possible_positives)
        self.n_predicted_positives.assign_add(predicted_positives)

    def result(self):
        precision = self.n_true_positives / (self.n_predicted_positives+epsilon)
        recall = self.n_true_positives / (self.n_possible_positives+epsilon)
        f_score = (1+self.beta**2) * (precision*recall) / (self.beta**2*precision+recall+epsilon)
        return f_score


def create_positive_counters(beta=1, threshold=0.5):

    def count_positives(y_true, y_pred):

        def count(x):
            return tf.reduce_sum(x)

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.where(y_pred>=threshold, 1., 0.)

        true_positives = count(y_true*y_pred)
        predicted_positives = count(y_pred)
        possible_positives = count(y_true)

        return true_positives, predicted_positives, possible_positives

    return count_positives


def create_fscore(beta=1, threshold=0.5):

    def f_score(y_true, y_pred):

        def count(x):
            return tf.reduce_sum(x)

        def recall(y_true, y_pred):
            true_positives = count(y_true*y_pred)
            possible_positives = count(y_true)
            recall = true_positives / (possible_positives+K.epsilon())
            return recall

        def precision(y_true, y_pred):
            true_positives = count(y_true*y_pred)
            predicted_positives = count(y_pred)
            precision = true_positives / (predicted_positives+K.epsilon())
            return precision

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.where(y_pred>=threshold, 1., 0.)

        P, R = precision(y_true, y_pred), recall(y_true, y_pred)
        return (1+beta**2)*(P*R)/(beta**2*P+R+K.epsilon())

    return f_score



