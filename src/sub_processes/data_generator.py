import numpy as np
import os
import sklearn as skl
from glob import glob
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):

    def __init__(self, data_root, opt, validate: bool=False):

        self.opt = opt
        self.data_root = data_root

        # list of files containing both word-embeddings and multi-labels
        if isinstance(self.data_root, str):
            self.files = glob(os.path.join(self.data_root, 'sample_*.npz'))
        elif isinstance(self.data_root, (list, tuple)):
            self.files = []
            for data_dir in self.data_root:
                self.files += glob(os.path.join(data_dir, 'sample_*.npz'))

        self.indices = np.array(list(range(len(self.files))))
        if not validate:
            self.batch_size = opt.batch_size
            self.shuffle = True
        else:
            self.batch_size = len(self.indices)
            self.shuffle = False

        self.on_epoch_end()

    def __len__(self):
        """ Denotes the number of batches per epoch """
        n_samples = len(self.files)
        return n_samples//self.batch_size + (0 if n_samples%self.batch_size==0 else 1)

    def __getitem__(self, index):
        """ Generate single batch of data """

        start_index = self.batch_size * index
        end_index = self.batch_size * (index+1)
        indices = self.indices[start_index:end_index]

        # Generate data
        emb_batch = []
        em_batch, sm_batch, pa_batch = [], [], []
        ay_batch, oy_batch, sy_batch, ctgy_batch, atty_batch = [], [], [], [], []

        for idx in indices:
            sample_file = self.files[idx]
            sample_data = np.load(sample_file)

            # Load embeddings
            emb_pad = np.zeros((self.opt.max_sentence_len, self.opt.embedding_dim))
            emb = sample_data['sent_emb']
            emb_pad[:emb.shape[0],:] = emb
            emb_batch += [emb_pad]

            # Load masks
            em_batch += [sample_data['sent_mask']]
            sm_batch += [sample_data['s_mask']]
            pa_batch += [sample_data['pos_att']]

            # Load labels
            ay = sample_data['a_y'].astype(float)
            oy = sample_data['o_y'].astype(float)
            sy = sample_data['s_y'].astype(float)
            catg_y = sample_data['catg_y'].astype(float)
            attr_y = sample_data['attr_y'].astype(float)
            ay_batch += [self.smooth_labels(ay) if self.opt.label_smoothing else ay]
            oy_batch += [self.smooth_labels(oy) if self.opt.label_smoothing else oy]
            sy_batch += [self.smooth_labels(sy) if self.opt.label_smoothing else sy]
            ctgy_batch += [self.smooth_labels(catg_y) if self.opt.label_smoothing else catg_y]
            atty_batch += [self.smooth_labels(attr_y) if self.opt.label_smoothing else attr_y]

        data_batch = [emb_batch, em_batch, sm_batch, pa_batch, 
                      ay_batch, oy_batch, sy_batch, ctgy_batch, atty_batch]
        data_batch = [np.array(d) for d in data_batch]
        return data_batch[:4], data_batch[4:]

    def smooth_labels(self, labels, factor=0.1):
        # smooth the labels
        labels *= (1 - factor)
        labels += (factor / labels.shape[-1])
        return labels

    def on_epoch_end(self):
        """ Update indices after each epoch """
        if self.shuffle:
            self.indices = skl.utils.shuffle(self.indices)



