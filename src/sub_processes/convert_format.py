import itertools
import json
import ntpath
import os
import pandas as pd
import re
import spacy
from glob import glob
from pprint import PrettyPrinter
from sentence_transformers import SentenceTransformer
from string import punctuation
from tqdm import tqdm

from src.utils import normalize_punctuations

# Download SpaCy models if needed
spacy_model = 'en_core_web_sm'
try:
    nlp = spacy.load(spacy_model)
except OSError:
    print("\n\n\n Downloading SpaCy model ...")
    spacy.cli.download(spacy_model)
    nlp = spacy.load(spacy_model)


# Define useful directories
working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
root_dir = os.path.dirname(working_dir)
data_dir = os.path.join(root_dir, 'dataset')


# Define useful variables
word_pattern = re.compile(r'\w+')
punct_pattern = re.compile(f"[{punctuation}]")
printer = PrettyPrinter(indent=4)

sentiment_mapper = {"pos": 1, "positive": 1,
                    "neg": 2, "negative": 2,
                    "neu": 3, "neutral": 3,
                    "conflict": 4, }

category_mapper = {"Arrival Experience": 1, 
                   "Room Services": 2, 
                   "Breakfast": 3,
                   "Dinning": 4,
                   "Bar & Lounge": 5,
                   "F&B Services": 6,
                   "Hotel Services": 7,
                   "Others": 8, }

attribute_mapper = {"Speed of check-in/out": 1,
                    "Booking accuracy": 2,
                    "Room availability": 3,
                    "Timeliness of service": 4,
                    "Loyalty Recognition": 5,
                    "Staff were responsive to my needs": 6,
                    "Accuracy of delivery of service": 7,
                    "Staff attitude / Staff anticipation": 8,
                    "Tools & Equipment": 9,
                    "Operational Hours": 10,
                    "Food quality": 11,
                    "Food variety": 12,
                    "Food temperature": 13,
                    "Beverage quality": 14,
                    "Price / Value": 15,
                    "Service Culture": 16,
                    "Problem Identification": 17,
                    "Service Recovery": 18,
                    "Hotel Facilities": 19,
                    "Location & Transportation": 20,
                    "Parking": 21,
                    "Disabled-Friendliness": 22,
                    "Room Cleanliness": 23,
                    "Room Amenities": 24,
                    "Room Condition": 25,
                    "Room Odour": 26,
                    "Noise Pollution": 27,
                    "Air-Condition": 28,
                    "Internet Connectivity": 29,
                    "Pest": 30,
                    "Shower / Bath Experience": 31,
                    "Planning": 32,
                    "Cleaning Process": 33,
                    "Others": 34, }

attribute_replacement = {"Queue": "Timeliness of service",
                         "Knowledge of staff": "Staff were responsive to my needs",
                         "Food Snack / Menu": "Food variety",
                         "Food presentation": "Food variety",
                         "Beverages quality": "Beverage quality",
                         "Beverage": "Beverage quality",
                         "Operations Hours": "Operational Hours",
                         "Primary Issues": "Service Culture",
                         "Track, Measure & Sustain": "Problem Identification",
                         "Transportation": "Location & Transportation",
                         "IHG Way of Clean 5-S Cleaning Process": "Cleaning Process",
                         "Cleaning tools": "Cleaning Process",
                         "Audits": "Others",
                         "PMM": "Others",
                         "PMM tools": "Others",                         
                         "Application of tools": "Others"}


def encode_words_location(text: str) -> dict:
    # print(f"\n{text}")
    text = normalize_punctuations(text)

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
    # printer.pprint(phrases_dict)

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
    # printer.pprint(words_dict)

    # Convert word dictionary to word dataframe --> easy comparison
    words_df = pd.DataFrame(data=words_dict).T
    words_df.rename(columns={0: 'offset_start',
                             1: 'offset_end',
                             2: 'word'}, inplace=True)
    # print(words_df)

    # Sentencize
    words_df['sentence_id'] = [0] * len(words_df)
    sentences = [sent.text for sent in nlp(text).sents]
    BoSs = [text.find(sent) for sent in sentences]
    for sentence_id, bos in enumerate(BoSs):
        if sentence_id == 0:
            continue
        words_df.loc[words_df.offset_start>=bos, 'sentence_id'] = sentence_id

    return words_dict, words_df


def decode_words_location(located_words: pd.DataFrame,
                          annotations: list or tuple) -> list:
    annotations = list(annotations)
    n_words = len(located_words)
    # located_words['word'] = located_words['word'].apply(lambda x: x.lower())

    # Assign all words as BACKGROUND
    located_words['aspect'] = [0] * n_words
    located_words['opinion'] = [0] * n_words
    located_words['sentiment'] = [0] * n_words
    located_words['category'] = [0] * n_words
    located_words['attribute'] = [0] * n_words

    # If no annotation, all words are considered background - class=0
    if len(annotations) < 1:
        return located_words

    for annotation in annotations:
        offset_start, offset_end, label = annotation[:3]

        # Assign all words in annotation as
        #       BEGIN code for Aspect & Opinion,
        #       polarity code for Sentiment,
        #       class code for Category & Attribute.
        annotation_locs = (located_words['offset_start']>=offset_start) & (located_words['offset_end']<=offset_end)
        if label == 'opinion':
            located_words.loc[annotation_locs, 'opinion'] = 1
        else:
            category, attribute, polarity = label.split('_-_')
            if attribute in attribute_replacement.keys():
                attribute = attribute_replacement[attribute]
            located_words.loc[annotation_locs, 'aspect'] = 1
            located_words.loc[annotation_locs, 'sentiment'] = sentiment_mapper[polarity]
            located_words.loc[annotation_locs, 'category'] = category_mapper[category]
            located_words.loc[annotation_locs, 'attribute'] = attribute_mapper[attribute]

        # Split Aspect & Opinion annotated words into BEGINNING and INSIDE
        for r_i in range(n_words-1, 0, -1):
            for col in ['opinion', 'aspect']:
                if located_words.loc[r_i, col] == 0:
                    continue
                if located_words.loc[r_i-1, col] == 1:
                    # if previous word is annotated as BEGINNING, flip current word to INSIDE
                    located_words.loc[r_i, col] = 2
    # print(located_words)
    return located_words


def window_slide(seq: list or tuple, window_size: int=2):
    seq_iter = iter(seq)
    seq_sliced = list(itertools.islice(seq_iter, window_size))
    if len(seq_sliced) == window_size:
        yield seq_sliced
    for seq_el in seq_iter:
        yield seq_sliced[1:] + [seq_el]


def process_word_level(annotation: dict,
                       lower_case: bool=False,
                       upper_case: bool=False,
                       capital_case: bool=False) -> pd.DataFrame:
    text = annotation['data']
    if lower_case:
        text = text.lower()
    elif upper_case:
        text = text.upper()
    elif capital_case:
        sentences = [sent.text for sent in nlp(text).sents]
        sentences = [sent.capitalize() for sent in sentences]
        text = '. '.join(sentences)
    _, located_words_df = encode_words_location(text)
    words_annotated = annotation['label'] if 'label' in annotation.keys() else []
    words_data = [a+[text[a[0]:a[1]]] for a in words_annotated]
    words_labeled = decode_words_location(located_words_df, words_data)
    words_labeled['doc_id'] = doc_id
    words_labeled.reset_index(inplace=True)
    words_labeled.rename(columns={'index': 'word_id'}, inplace=True)
    return words_labeled[words_labeled.word!=' ']


def process_token_level(words_labeled: pd.DataFrame) -> pd.DataFrame:
    words_df = words_labeled[['word_id', 'word']]
    words = words_df.word.values.tolist()
    text = ' '.join(words)
    tokens = tokenizer.tokenize(text)
    tokens_df = pd.DataFrame(columns=['token', 'word', 'word_id'])
    word_offset, word_id, token_id = 0, 0, 0
    word = words.pop(0)
    for token_id, token in enumerate(tokens):
        token_ = token[2:] if token.startswith('##') else token
        if token_ == word[word_offset:word_offset+len(token_)]:
            tokens_df.loc[token_id] = [token, word, word_id]
            word_offset += len(token_)
        else:
            print(f"\n\t{doc_id}\t{token_}\t{word[:len(token_)]}")
        if word_offset >= len(word) and len(words) > 0:
            word = words.pop(0)
            word_id += 1
            word_offset = 0
    tokens_labeled = tokens_df.merge(words_labeled, on=['word', 'word_id'], how='inner')
    return tokens_labeled


if __name__ == "__main__":

    # Load tokenizer
    print(f"\n\n\nLoading Tokenizer ...")
    sbert_version = 'distilUSE'
    sbert_dir = os.path.join(root_dir, 'artefacts', sbert_version)
    embedder = SentenceTransformer(sbert_dir)
    tokenizer = embedder.tokenizer

    # Processing
    label_cols = ['document', 'num_sentences', 'token', 'target', 'opinion', 'target_polarity', 'category', 'attribute']
    labels_df = pd.DataFrame(columns=label_cols)
    df_cols = ['token', 'aspect', 'opinion', 'sentiment', 'category', 'attribute']
    r_i = 0
    max_length = -1
    files = glob(os.path.join(data_dir, 'annotation*.json'))
    for fn in files:
        print(f"\n\n\nProcessing {ntpath.split(fn)[-1]} ...")
        with open(fn, 'r', encoding='utf-8') as f_reader:
            annotations = f_reader.readlines()

        # Process line-by-line
        for doc_id, annotation_unparsed in tqdm(enumerate(annotations)):
            annotation = json.loads(annotation_unparsed)
            # print(json.dumps(annotation, indent=4, sort_keys=True))

            # Data Augmentation
            if annotation['data'].islower():
                augment_cases = ({}, {'upper_case': True}, {'capital_case': True})
            elif annotation['data'].isupper():
                augment_cases = ({}, {'lower_case': True}, {'capital_case': True})
            else:
                augment_cases = ({}, {'lower_case': True}, {'upper_case': True})

            for case_dict in augment_cases:
                # process word-level
                try:
                    words_labeled = process_word_level(annotation, **case_dict)
                except Exception as e:
                    print(f"[process word-level] {e}: {annotation['data']}")
                    continue

                # process token-level
                tokens_labeled = process_token_level(words_labeled)

                # Sentencize and Group
                sentence_ids = tokens_labeled.sentence_id.unique()
                for N in range(len(sentence_ids)):
                    for ws in window_slide(sentence_ids, N+1):
                        tokens_in_sentences = tokens_labeled[tokens_labeled.sentence_id.isin(ws)]
                        words_df = tokens_labeled[tokens_labeled.sentence_id.isin(ws)][['word', 'word_id']].drop_duplicates()
                        doc = ' '.join(words_df.word.values.tolist())

                        # Feed label into DataFrame
                        labels_df.loc[r_i] = [doc, len(ws)] + [tokens_in_sentences[col].values.tolist() for col in df_cols]
                        r_i += 1

                        # Measure max sequence length
                        tokens_measured = tokens_labeled.token.values.tolist()
                        max_length = max(len(tokens_measured), max_length)

    labels_df.drop_duplicates(subset=['document'], inplace=True)
    labels_df.to_csv(os.path.join(data_dir, f"annotations_preprocessed.csv"), index=False)
    print(f"\n\nMax sequence length = {max_length}")
    print(labels_df.num_sentences.value_counts())

    # Reproduce vocabulary
    embedding_dir = glob(os.path.join(sbert_dir, '0_*'))[0]
    vocab_path = os.path.join(embedding_dir, 'vocab.txt')
    with open(vocab_path, 'r', encoding='utf-8') as f_reader:
        vocab_list = f_reader.readlines()
    vocab_dict = {w[:-1]: w_id for w_id, w in enumerate(vocab_list)}
    with open(os.path.join(data_dir, 'vocab.dict'), 'w', encoding='utf-8') as f_writer:
        f_writer.write(json.dumps(vocab_dict, indent=4))


    ##########################################
    #   Executive Time on Local Machine:     #
    #       619it [01:06,  9.31it/s]         #
    ##########################################



