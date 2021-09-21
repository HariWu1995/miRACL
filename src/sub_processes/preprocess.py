import os
import numpy as np
import pandas as pd
from ast import literal_eval
# from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm as print_progress

from src.models.encoder import Encoder
from src.sub_processes.convert_format import attribute_mapper, category_mapper


# Define useful directories
working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_dir = os.path.dirname(working_dir)
data_dir = os.path.join(root_dir, 'dataset')


# Define useful variables
val_size, test_size = 512, 1024
random_seed = 4_10_20
MAX_SEQ_LEN = 256
N_CATEGORIES = len(category_mapper.keys())
N_ATTRIBUTES = len(attribute_mapper.keys())


def vectorize_data(embeddings: np.array,
                   aspects: list,
                   opinions: list,
                   sentiments: list,
                   categories: list,
                   attributes: list,
                   is_training: bool=False):
    
    source_data, source_mask, sentiment_mask, position_attention, \
    aspect_y, opinion_y, sentiment_y, category_y, attribute_y = [], [], [], [], [], [], [], [], []

    for Es, As, Os, Ss, Catgs, Attrs in zip(embeddings, aspects, opinions, sentiments, categories, attributes):

        L, D = Es.shape[:2]
        n_pads = MAX_SEQ_LEN - L

        # Check length between input and labels
        for v, v_name in zip([As, Os, Ss], ['aspect', 'opinion', 'sentiment']):
            assert L==len(v), f"length of {v_name} is different from number of tokens"

        # Read Embedding Input
        e_padded = np.zeros(shape=(MAX_SEQ_LEN, D))
        e_padded[:L, :] = Es
        source_data.append(e_padded)
        source_mask.append([1.]*L + [0.]*n_pads)

        # Read labels
        aspect_label = to_categorical(As, num_classes=3, dtype='int').tolist()
        opinion_label = to_categorical(Os, num_classes=3, dtype='int').tolist()
        category_label = to_categorical(Catgs, num_classes=N_CATEGORIES, dtype='int').tolist()
        attribute_label = to_categorical(Attrs, num_classes=N_ATTRIBUTES, dtype='int').tolist()

        sentiment_label, sentiment_m = [], []
        for l in Ss:
            l = int(l)
            if 1 <= l <= 3:
                sentiment_label.append(to_categorical(l-1, num_classes=3, dtype='int').tolist())
                sentiment_m.append(1.)
            else:
                sentiment_label.append([0, 0, 0])
                sentiment_m.append(0. if is_training else 1.)

        # 0-padding for consistent length of input
        aspect_y.append(aspect_label + [[0]*3]*n_pads)
        opinion_y.append(opinion_label + [[0]*3]*n_pads)
        category_y.append(category_label + [[0]*N_CATEGORIES]*n_pads)
        attribute_y.append(attribute_label + [[0]*N_ATTRIBUTES]*n_pads)
        sentiment_y.append(sentiment_label + [[0]*3]*n_pads)
        sentiment_mask.append(sentiment_m + [0.]*n_pads)
        position_attention.append(position_matrix(L))

    # Ensure correct input format
    aspect_y = [np.array(ay[:MAX_SEQ_LEN]).astype(np.int) for ay in aspect_y]
    opinion_y = [np.array(oy[:MAX_SEQ_LEN]).astype(np.int) for oy in opinion_y]
    category_y = [np.array(cy[:MAX_SEQ_LEN]).astype(np.int) for cy in category_y]
    attribute_y = [np.array(ay[:MAX_SEQ_LEN]).astype(np.int) for ay in attribute_y]
    sentiment_y = [np.array(sa[:MAX_SEQ_LEN]).astype(np.int) for sa in sentiment_y]
    source_mask = [np.array(xm[:MAX_SEQ_LEN]).astype(np.float) for xm in source_mask]
    sentiment_mask = [np.array(sm[:MAX_SEQ_LEN]).astype(np.float) for sm in sentiment_mask]

    return source_data, source_mask, sentiment_mask, position_attention, \
           aspect_y, opinion_y, sentiment_y, category_y, attribute_y


def position_matrix(sen_len: int) -> np.array:
    A = np.zeros([MAX_SEQ_LEN, MAX_SEQ_LEN], dtype=np.float32)
    for i in range(sen_len):
        for j in range(sen_len):
            A[i][j] = 0. if i==j else ( 1 / (np.log2(2+abs(i-j))) )
    return A


if __name__ == "__main__":

    # Read data
    labels_df = pd.read_csv(os.path.join(data_dir, f"annotations_preprocessed.csv"))
    # labels_df = labels_df[labels_df.num_sentences==1]

    # Load Encoder
    print(f"\n\n\nLoading Encoder ...")
    sbert_version = 'distilUSE'
    sbert_dir = os.path.join(root_dir, 'artefacts', sbert_version)
    encoder = Encoder(sbert_dir)

    # Train-Validation-Test Split
    dataset = dict()
    dataset['train'], dataset_valtest = train_test_split(labels_df, test_size=test_size+val_size, random_state=random_seed)
    dataset['val'], dataset['test'] = train_test_split(dataset_valtest, test_size=test_size, random_state=random_seed)

    # Preprocessing --> save .npz file
    max_num_tokens = -1
    for subset_name, subset in dataset.items():
        print(f"\n\nPreprocessing {subset_name}-set ...")
        subset_dir = os.path.join(data_dir, subset_name)
        if not os.path.isdir(subset_dir):
            os.makedirs(subset_dir)

        # Get sentences and labels
        docs = subset.document.values.tolist()
        tokens = subset.token.values.tolist()
        aspects = subset.target.values.tolist()
        opinions = subset.opinion.values.tolist()
        categories = subset.category.values.tolist()
        attributes = subset.attribute.values.tolist()
        sentiments = subset.target_polarity.values.tolist()

        # Convert str to list
        tokens = [literal_eval(t) for t in tokens]
        aspects = [literal_eval(a) for a in aspects]
        opinions = [literal_eval(o) for o in opinions]
        categories = [literal_eval(c) for c in categories]
        attributes = [literal_eval(a) for a in attributes]
        sentiments = [literal_eval(s) for s in sentiments]

        # Arguments
        batch_size = 32
        n_samples = len(docs)
        n_batches = n_samples//batch_size + (0 if n_samples%batch_size==0 else 1)

        # Preprocess by batch
        for b_idx in print_progress(range(n_batches)):

            # Get batch data
            if b_idx != n_batches-1:
                b_samples = docs[b_idx*batch_size:(b_idx+1)*batch_size]
                b_tokens = tokens[b_idx*batch_size:(b_idx+1)*batch_size]
                b_aspects = aspects[b_idx*batch_size:(b_idx+1)*batch_size]
                b_opinions = opinions[b_idx*batch_size:(b_idx+1)*batch_size]
                b_categories = categories[b_idx*batch_size:(b_idx+1)*batch_size]
                b_attributes = attributes[b_idx*batch_size:(b_idx+1)*batch_size]
                b_sentiments = sentiments[b_idx*batch_size:(b_idx+1)*batch_size]
            else:
                b_samples = docs[b_idx*batch_size:]
                b_tokens = tokens[b_idx*batch_size:]
                b_aspects = aspects[b_idx*batch_size:]
                b_opinions = opinions[b_idx*batch_size:]
                b_categories = categories[b_idx*batch_size:]
                b_attributes = attributes[b_idx*batch_size:]
                b_sentiments = sentiments[b_idx*batch_size:]

            # Apply sentence-BERT for word embeddings
            b_embeddings = encoder.encode(b_samples,
                                          batch_size=batch_size,
                                          output_value='token_embeddings',
                                          show_progress_bar=False)

            # Remove embeddings of tokens [CLS] and [SEP] - those for other tasks
            b_embeddings = [e[1:-1, :] for e in b_embeddings]

            # Save
            b_data = vectorize_data(b_embeddings, b_aspects, b_opinions, b_sentiments, b_categories, b_attributes,
                                    is_training=True if 'train' in subset_name else False)
            for s_idx, (sent_emb, sent_mask, s_mask, pos_att, a_y, o_y, s_y, catg_y, attr_y) in enumerate(zip(*b_data)):
                max_num_tokens = max(sent_emb.shape[0], max_num_tokens)
                np.savez_compressed(
                    file=os.path.join(subset_dir, f'sample_{b_idx*batch_size+s_idx:07d}.npz'),
                    sent_emb=sent_emb, sent_mask=sent_mask, s_mask=s_mask, pos_att=pos_att, 
                    a_y=a_y, o_y=o_y, s_y=s_y, catg_y=catg_y, attr_y=attr_y
                )

    print(f"\n\nMax number of tokens: {max_num_tokens}")


    ######################################################
    #   Executive Time on Local Machine:                 #
    #       Train-set: 347/347 [17:17<00:00,  2.99s/it]  #
    #         Val-set:  16/ 16 [00:53<00:00,  3.32s/it]  #
    #        Test-set:  32/ 32 [01:37<00:00,  3.05s/it]  #
    ######################################################



