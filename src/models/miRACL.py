"""
miRACL - MultI-label Relation-Aware Collaborative Learning
            for Unified Aspect-based Sentiment Analysis

    # Sentence Representation
        A. Word Embeddings - Pretrained distil-USE (multi-lingual sentence embeddings)

        B. Features:
            0. Sharing Feature: Dropout -> CNN -> Dropout

            1. Aspect Extraction [AE]: CNN

            2. Opinion Extraction [OE]: CNN

            3. Sentiment Classification [SC]: CNN

        C. Relations:
            1. R1 - relation between AE and OE: Attention

            2. R2 - relation between SC and R1: Attention

            3. R3 - relation between SC and OE: Attention

            4. R4 - relation between SC and AE: Attention

            5. Opinion Propagation: SoftMax -> Clip

    # Predictions + Loss Function
        A. Aspect: Fully-Connected -> SoftMax -> Cross-Entropy

        B. Opinion: Fully-Connected -> SoftMax -> Cross-Entropy

        C. Sentiment: Fully-Connected -> SoftMax -> Cross-Entropy

        D. Category: Fully-Connected -> SoftMax -> Cross-Entropy

        E. Attribute: Fully-Connected -> SoftMax -> Cross-Entropy

        F. Loss = Weighted_Sum(Aspect_Loss, Opinion_Loss, Sentiment_Loss, Category_Loss, Attribute_Loss, Regularization_Loss)
"""
import logging
import ntpath
import numpy as np
import os
import pickle
import tensorflow as tf
import time
from glob import glob

if not tf.executing_eagerly():
    tf.enable_eager_execution()
# tf.config.run_functions_eagerly(True)

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dropout, Concatenate, Lambda, Softmax, )
try:
    from tensorflow.keras.activations import softmax, sigmoid
    from tensorflow.keras.initializers import Identity, GlorotNormal, GlorotUniform
    from tensorflow.keras.optimizers import Adam, Nadam, Adagrad, Adadelta, RMSprop, SGD
    from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, LearningRateScheduler, EarlyStopping, ModelCheckpoint, Callback
    from tensorflow.keras.losses import CategoricalCrossentropy
    from tensorflow.keras.utils import plot_model
except ImportError as e:
    pass

from src.utils import log_summary, plot_history
from src.models.racl.layers import dropoutize_embeddings, DropBlock2D, L2Norm, RACL_Block, ExpandDim, ReduceDim
from src.sub_processes.losses import RACL_losses
from src.sub_processes.metrics import evaluate_absa, evaluate_multilists
from src.sub_processes.optimizers import get_optimizer
from src.sub_processes.lr_schedulers import CyclicLR
from src.sub_processes.data_generator import DataGenerator


class MIRACL(object):

    def __init__(self, opt):
        self.opt = opt
        self.mode = 'train' if opt.is_training else 'predict'
        if opt.random_type == 'uniform':
            self.initializer = GlorotUniform(seed=opt.random_seed)
        else:
            self.initializer = GlorotNormal(seed=opt.random_seed)

        if opt.is_training:
            # Build logger
            log_dir = opt.logs_path
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            filename = os.path.join(log_dir, f'{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}.txt')
            self.logger = logging.getLogger(filename)
            self.logger.setLevel(logging.DEBUG)
            # self.logger.propagate = False
            self.logger.addHandler(logging.StreamHandler())
            self.logger.addHandler(logging.FileHandler(filename, 'a'))

            # Log hyper-parameters
            info = ''
            for arg in vars(opt):
                info += ('>>> {0}: {1}\n'.format(arg, getattr(opt, arg)))
            self.logger.info('{:-^80}\n{}\n'.format('Parameters', info))

            # Build checkpoint & tensorboard
            self.ckpt_dir = opt.ckpt_path
            self.board_dir = os.path.join(opt.output_path, "tensorboard")
            self.viz_dir = os.path.join(opt.output_path, "visualization")
            for dir_ in [self.ckpt_dir, self.board_dir, self.viz_dir]:
                if not os.path.isdir(dir_):
                    os.makedirs(dir_)

        # Build model
        inputs, embeddings, position_att, token_mask, sentiment_mask = self.build_input_block()
        predictions = list(self.build_RACL_block(embeddings, position_att, token_mask))

        if opt.is_training or opt.is_evaluating:
            model_inputs = [inputs, token_mask, sentiment_mask, position_att]
            model_outputs = predictions + [token_mask, sentiment_mask]
            self.model = CustomModel(inputs=model_inputs, outputs=model_outputs, name='miRACL')

            if opt.is_training:
                model_summary = log_summary(self.model)
                self.logger.info(model_summary)
                self.visualize_architecture()
        else:
            predictions_as_prob = self.build_output_block(predictions)
            self.model = CustomModel(inputs=[inputs, token_mask, position_att],
                                     outputs=predictions_as_prob, name='miRACL')
            self.model.summary()

    def visualize_architecture(self):
        plot_model(self.model, to_file=f'{self.opt.model}_{self.mode}.png', dpi=128, show_shapes=True, show_layer_names=True)

    def build_input_block(self):
        inputs, embeddings = dropoutize_embeddings(self.opt)
        inputs._name = 'embeddings_concat'

        # Inputs for Masking
        position_att = Input(shape=(self.opt.max_sentence_len, self.opt.max_sentence_len), name='position_att')
        token_mask = Input(shape=(self.opt.max_sentence_len,), name='token_mask')
        sentiment_mask = Input(shape=(self.opt.max_sentence_len,), name='sentiment_mask')
        return inputs, embeddings, position_att, token_mask, sentiment_mask

    def build_RACL_block(self, embeddings, position_att, token_mask):
        # Preprocessing
        inputs = Dropout(rate=1-self.opt.keep_prob_1, name='inputs_dropout')(embeddings)

        # Shared Features
        conv_args = {'kernel_size': 1, 'strides': 1, 'padding': 'same', 'activation': 'relu', }
        Feature_Extractor = Conv1D(filters=self.opt.embedding_dim, name='shared_features', **conv_args)
        shared_features = Feature_Extractor(inputs)
        shared_features = Dropout(rate=1-self.opt.keep_prob_1, name='shared_features_dropout')(shared_features)

        # Define repeatable layers in RACL interactions
        DropBlock_aspect = DropBlock2D(keep_prob=self.opt.keep_prob_2, block_size=3, name='DropBlock2D_aspect')
        DropBlock_opinion = DropBlock2D(keep_prob=self.opt.keep_prob_2, block_size=3, name='DropBlock2D_opinion')
        DropBlock_context = DropBlock2D(keep_prob=self.opt.keep_prob_2, block_size=3, name='DropBlock2D_context')

        L2Normalize = L2Norm()
        Tile = Lambda(lambda x: tf.tile(tf.expand_dims(x, axis=1), [1, self.opt.max_sentence_len, 1]), name='Tiler-in-RACL')

        # We found that the SC task is more difficult than the AE and OE tasks.
        # Hence, we augment it with a memory-like mechanism by updating the aspect query with the retrieved contexts.
        # For more details about the memory network, refer to
        #       https://www.aclweb.org/anthology/D16-1021/ .
        aspect_inputs, opinion_inputs, context_inputs = [shared_features], [shared_features], [shared_features]
        aspect_preds, opinion_preds, sentiment_preds, category_preds, attribute_preds = [], [], [], [], []
        context_queries = [shared_features]

        conv_args['kernel_size'] = self.opt.kernel_size
        classifier_args = dict()
        for clss, n_clss in zip(['aspect', 'opinion', 'sentiment', 'category', 'attribute'], 
                                [3, 3, self.opt.n_sentiments, self.opt.n_categories, self.opt.n_attributes]):
            classifier_args[clss] = {'units': n_clss, 
                                     'activation': 'softmax',
                                     'kernel_initializer': self.initializer, }

        for interact_i in range(self.opt.n_interactions):
            racl_block = RACL_Block(self.opt, L2Normalize, [DropBlock_aspect, DropBlock_opinion, DropBlock_context], Tile,
                                    conv_args, classifier_args, block_id=interact_i)
            output_preds, output_interacts = racl_block([aspect_inputs[-1], opinion_inputs[-1], context_inputs[-1], context_queries[-1], token_mask, position_att])
            aspect_pred, opinion_pred, sentiment_pred, category_pred, attribute_pred = output_preds
            aspect_interact, opinion_interact, context_interact, context_conv = output_interacts

            # Stacking
            aspect_preds.append(ExpandDim(axis=-1, name=f'aspect_pred-{interact_i}')(aspect_pred))
            opinion_preds.append(ExpandDim(axis=-1, name=f'opinion_pred-{interact_i}')(opinion_pred))
            sentiment_preds.append(ExpandDim(axis=-1, name=f'sentiment_pred-{interact_i}')(sentiment_pred))
            category_preds.append(ExpandDim(axis=-1, name=f'category_pred-{interact_i}')(category_pred))
            attribute_preds.append(ExpandDim(axis=-1, name=f'attribute_pred-{interact_i}')(attribute_pred))

            aspect_inputs.append(aspect_interact)
            opinion_inputs.append(opinion_interact)
            context_inputs.append(context_conv)
            context_queries.append(context_interact) # update query

        # Multi-layer Short-cut
        aspect_preds = Concatenate(axis=-1, name='aspect_preds')(aspect_preds)
        opinion_preds = Concatenate(axis=-1, name='opinion_preds')(opinion_preds)
        sentiment_preds = Concatenate(axis=-1, name='sentiment_preds')(sentiment_preds)
        category_preds = Concatenate(axis=-1, name='category_preds')(category_preds)
        attribute_preds = Concatenate(axis=-1, name='attribute_preds')(attribute_preds)
        aspect_pred = ReduceDim('mean', axis=-1, name='AE_pred')(aspect_preds)
        opinion_pred = ReduceDim('mean', axis=-1, name='OE_pred')(opinion_preds)
        sentiment_pred = ReduceDim('mean', axis=-1, name='SC_pred')(sentiment_preds)
        category_pred = ReduceDim('mean', axis=-1, name='CC_pred')(category_preds)
        attribute_pred = ReduceDim('mean', axis=-1, name='AC_pred')(attribute_preds)
        return aspect_pred, opinion_pred, sentiment_pred, category_pred, attribute_pred

    def build_output_block(self, preds):
        aspect_pred, opinion_pred, sentiment_pred, category_pred, attribute_pred = preds

        # Scale probability
        aspect_prob = Softmax(axis=-1, name='aspect_prob')(aspect_pred)
        opinion_prob = Softmax(axis=-1, name='opinion_prob')(opinion_pred)
        sentiment_prob = Softmax(axis=-1, name='sentiment_prob')(sentiment_pred)
        category_prob = Softmax(axis=-1, name='category_prob')(category_pred)
        attribute_prob = Softmax(axis=-1, name='attribute_prob')(attribute_pred)
        return aspect_prob, opinion_prob, sentiment_prob, category_prob, attribute_prob

    def train(self):

        # Load generators
        train_set = DataGenerator(self.opt.train_path, self.opt, validate=False)
        val_set = DataGenerator(self.opt.val_path, self.opt, validate=True)
        test_set = DataGenerator(self.opt.test_path, self.opt, validate=True)
        n_trains, n_vals, n_tests = len(train_set), len(val_set), len(test_set)

        ################################
        #     Training Procedure       #
        ################################
        Evaluator = ABSA_Evaluation(val_set[0], self.logger, opt=self.opt, include_opinion=self.opt.include_opinion)
        train_arguments = {
            'x': train_set,
            'steps_per_epoch': n_trains,
            'validation_data': val_set,
            'validation_steps': n_vals,
            'verbose': 1,
            'callbacks': [
                # ReduceLROnPlateau(monitor='val_loss', factor=0.69, patience=5, min_lr=1e-7, verbose=1),
                CyclicLR(mode='exponential', base_lr=self.opt.lr//169, max_lr=self.opt.lr, step_size=n_trains*2),
                # TensorBoard(self.board_dir),
                ModelCheckpoint(os.path.join(self.opt.ckpt_path, 'miRACL-epoch={epoch:03d}.h5'), save_weights_only=True, monitor='loss', verbose=1),
                Evaluator,
                # EarlyStopping(monitor="val_loss", patience=11, restore_best_weights=True, verbose=1)
            ]
        }
        self.model.set_opt(self.opt)
        self.model.compile(optimizer=get_optimizer(self.opt.optimizer, learning_rate=self.opt.lr))

        phases = ['all', 'opinion', 'aspect', 'sentiment', 'all']
        epochs = [p*self.opt.n_epochs for p in range(len(phases)+1)]
        histories = []

        for l in range(self.opt.n_loops):
            self.logger.info(f"\n\tLoop {l+1:03d} / {self.opt.n_loops:03d}")
            for p_i, phase in enumerate(phases):
                self.logger.info(f"\n\t\tPhase {p_i+1}: Training {phase.upper()} layers ...")
                history = self.train_per_phase(initial_epoch=l*self.opt.n_epochs*len(phases)+epochs[p_i],
                                                      epochs=l*self.opt.n_epochs*len(phases)+epochs[p_i+1],
                                             train_arguments=train_arguments,
                                                       phase=phase)
                histories.append(history)

                # Update weights for losses
                self.logger.info(f"\n\t\tPhase {p_i+1}: Updating loss weights ...")
                if p_i >= len(phases)-1:
                    if p_i == len(phases) and l == self.opt.n_loops-1:
                        continue
                    f1_o, f1_a, _, f1_s, _ = self.evaluate(test_set=val_set, RACL_only=True)
                    scores = np.array([f1_a, f1_o, f1_s], dtype=float)
                    weights = 1 / (scores+K.epsilon())
                    weights /= np.min(weights)
                    weights = np.clip(weights, 1., 16.9)
                else:
                    next_phase = phases[p_i+1]
                    if next_phase == 'aspect':
                        weights = [3.69, 1.00, 1.00]
                    elif next_phase == 'opinion':
                        weights = [1.00, 3.69, 1.00]
                    elif next_phase == 'sentiment':
                        weights = [1.00, 1.00, 1.69]
                    else:
                        weights = [1.00, 1.00, 1.00]

                self.opt.aspect_weight = weights[0]
                self.opt.opinion_weight = weights[1]
                self.opt.sentiment_weight = weights[2]
                self.logger.info(f"\n\t\t\t aspect_weight = {weights[0]} \n\t\t\t opinion_weight = {weights[1]} \n\t\t\t sentiment_weight = {weights[2]}")
 
            # Save best weights per phase
            ckpt_ids_to_keep = [Evaluator.min_loss_index, Evaluator.max_score_ABSA_index, Evaluator.max_score_miRACL_index]
            for ckpt_id, ckpt_type in zip(ckpt_ids_to_keep, ['loss', 'score_ABSA', 'score_miRACL']):
                model_ckpt = os.path.join(self.ckpt_dir, f'miRACL-epoch={ckpt_id:03d}.h5')
                self.model.load_weights(model_ckpt)
                self.model.save_weights(os.path.join(self.ckpt_dir, f'miRACL-best-{ckpt_type}-loop={l+1}.h5'))

            # Clean epoch weights
            ckpt_ids_to_keep = [Evaluator.min_loss_index, Evaluator.max_score_ABSA_index, Evaluator.max_score_miRACL_index]
            ckpt_ids_to_keep = list(set(ckpt_ids_to_keep))
            for ckpt_file in glob(os.path.join(self.ckpt_dir, 'miRACL-epoch=*.h5')):
                ckpt_id = int(ntpath.basename(ckpt_file)[11:14])
                if ckpt_id in ckpt_ids_to_keep:
                    continue
                if os.path.isfile(ckpt_file):
                    os.remove(ckpt_file)

        # Visualization
        try:
            history_fig = plot_history(histories)
            history_fig.savefig(os.path.join(self.viz_dir, 'training_history.png'))

            for t_i, train_history in enumerate(histories):
                with open(f'train_history_{t_i}.hst', 'wb') as f_writer:
                    pickle.dump(train_history.history, f_writer)
        except Exception:
            pass

        # Testing Process
        self.logger.info('\n\t Testing')
        for ckpt_file in sorted(glob(os.path.join(self.ckpt_dir, 'miRACL-best-*.h5'))):
            scores = self.evaluate(model_ckpt=ckpt_file, test_set=test_set)
            self.logger.info(f'\n\t Prediction by {ntpath.basename(ckpt_file)}')
            self.logger.info(f'\t\t opinion_f1={scores[0]:.7f} \n\t\t aspect_f1={scores[1]:.7f} \n\t\t sentiment_acc={scores[2]:.7f} \n\t\t sentiment_f1={scores[3]:.7f} \n\t\t ABSA_f1={scores[4]:.7f}')
            self.logger.info(f'\t\t category_f1={scores[5]:.7f} \n\t\t category_acc={scores[6]:.7f} \n\t\t attribute_f1={scores[7]:.7f} \n\t\t attribute_acc={scores[8]:.7f} \n\t\t miRACL_f1={scores[9]:.7f}')

    def train_aspect(self, initial_epoch: int, epochs: int, train_arguments: dict):
        for layer in self.model.layers:
            if 'aspect' in layer.name.lower():
                layer.trainable = True
                self.logger.info(f"\t\t\t{layer.name}")
            else:
                layer.trainable = False
        history = self.model.fit(initial_epoch=initial_epoch, epochs=epochs, **train_arguments)
        return history

    def train_opinion(self, initial_epoch: int, epochs: int, train_arguments: dict):
        for layer in self.model.layers:
            if 'opinion' in layer.name.lower():
                layer.trainable = True
                self.logger.info(f"\t\t\t{layer.name}")
            else:
                layer.trainable = False
        history = self.model.fit(initial_epoch=initial_epoch, epochs=epochs, **train_arguments)
        return history

    def train_sentiment(self, initial_epoch: int, epochs: int, train_arguments: dict):
        for layer in self.model.layers:
            if any(ss in layer.name.lower() for ss in ['sentiment', 'context']):
                layer.trainable = True
                self.logger.info(f"\t\t\t{layer.name}")
            else:
                layer.trainable = False
        history = self.model.fit(initial_epoch=initial_epoch, epochs=epochs, **train_arguments)
        return history

    def train_all(self, initial_epoch: int, epochs: int, train_arguments: dict):
        for layer in self.model.layers:
            layer.trainable = True
        history = self.model.fit(initial_epoch=initial_epoch, epochs=epochs, **train_arguments)
        return history

    def train_per_phase(self, initial_epoch: int, epochs: int, train_arguments: dict, phase: str='all'):

        phase = phase.lower()
        if phase == 'embedding':
            history = self.train_embedding(initial_epoch, epochs, train_arguments)
        elif phase == 'aspect':
            history = self.train_aspect(initial_epoch, epochs, train_arguments)
        elif phase == 'opinion':
            history = self.train_opinion(initial_epoch, epochs, train_arguments)
        elif phase == 'sentiment':
            history = self.train_sentiment(initial_epoch, epochs, train_arguments)
        else:
            history = self.train_all(initial_epoch, epochs, train_arguments)
        return history

    def evaluate(self, model_ckpt='', test_set=None, RACL_only: bool=False):
        # Load generator
        if not isinstance(test_set, DataGenerator):
            test_set = DataGenerator(self.opt.test_path, self.opt, validate=True)

        # Load weights
        if model_ckpt != '' and os.path.isfile(model_ckpt):
            self.model.load_weights(model_ckpt)

        # Evaluate
        Xs, Ys_true = test_set[0]
        *Ys_pred, token_mask, _ = self.model.predict(Xs)
        scores_RACL = evaluate_absa(Ys_true[0], Ys_pred[0],
                                    Ys_true[1], Ys_pred[1],
                                    Ys_true[2], Ys_pred[2],
                                    token_mask, include_opinion=self.opt.include_opinion)
        if RACL_only:
            return scores_RACL

        scores_MI = evaluate_multilists(Ys_true[3:], Ys_pred[3:], token_mask)
        score_miRACL = (scores_RACL[4] + (scores_MI[0]+scores_MI[2])/2) / 2
        return scores_RACL + scores_MI + [score_miRACL]

    def predict(self, sentence, token_mask, position_att):
        """
        Return:
            ae_pred, oe_pred, sc_pred, ctgr_pred, attr_pred
        """
        Ys_pred = self.model.predict([sentence, token_mask, position_att])
        return Ys_pred 

    def load_weights(self, weights_path):
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"weights_path:\n\t{weights_path}\ndoesn't exist!")
        try:
            self.model.load_weights(weights_path)
        except Exception as e:
            print(e)


class CustomModel(Model):

    def set_opt(self, opt):
        self.opt = opt

    def train_step(self, data):
        Xs, Ys_true = data

        with tf.GradientTape() as tape:
            # Forward pass
            *Ys_pred, token_mask, sentiment_mask = self(Xs, training=True)

            # Compute the loss value. Loss function is configured in `compile()`.
            losses = RACL_losses(Ys_true, Ys_pred, [token_mask, sentiment_mask], self.opt)

        # Backward progagation - Compute gradients & Update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(losses[0], trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {
            'OE_loss': losses[1], 'AE_loss': losses[2], 'SC_loss': losses[3],
            'CC_loss': losses[4], 'AC_loss': losses[5],
            'Reg_cost': losses[6], 'loss': losses[0], 'lr': self.optimizer.learning_rate,
        }

    def test_step(self, data):
        # Unpack the data
        Xs, Ys_true = data

        # Compute predictions
        *Ys_pred, token_mask, sentiment_mask = self(Xs, training=False)

        # Compute the loss value
        losses = RACL_losses(Ys_true, Ys_pred, [token_mask, sentiment_mask], self.opt)
        return {
            'OE_loss': losses[1], 'AE_loss': losses[2], 'SC_loss': losses[3],
            'CC_loss': losses[4], 'AC_loss': losses[5],
            'Reg_cost': losses[6], 'loss': losses[0],
        }


class ABSA_Evaluation(Callback):

    def __init__(self, validation_data, logger, opt, include_opinion: bool=True, threshold: float=0.2, name='ABSA_scores', **kwargs):
        super().__init__(**kwargs) # handle base args (e.g. dtype)
        self._name = name
        self.Xs, self.Ys_true = validation_data
        self.opt = opt
        self.logger = logger
        self.include_opinion = include_opinion
        self.threshold = threshold
        self.records = {
            'opinion_f1': [], 'OE_loss': [],
            'aspect_f1': [], 'AE_loss': [],
            'sentiment_acc': [], 'sentiment_f1': [], 'SC_loss': [],
            'ABSA_f1': [], 'miRACL_f1': [], 
            'category_acc': [], 'category_f1': [], 'CC_loss': [], 
            'attribute_acc': [], 'attribute_f1': [], 'AC_loss': [], 
            'Reg_cost': [], 'total_loss': [], 
        }

    def on_train_begin(self, logs={}):
        ...

    def on_epoch_end(self, epoch, logs={}):
        start = time.time()

        # Forward pass
        *Ys_pred, token_mask, sentiment_mask = self.model(self.Xs, training=False)

        # Compute losses
        losses = RACL_losses(self.Ys_true, Ys_pred, [token_mask, sentiment_mask], self.opt)

        # Evaluate
        scores_RACL = evaluate_absa(self.Ys_true[0], Ys_pred[0],
                                    self.Ys_true[1], Ys_pred[1],
                                    self.Ys_true[2], Ys_pred[2],
                                    token_mask, self.include_opinion)
        scores_MI = evaluate_multilists(self.Ys_true[3:], Ys_pred[3:], token_mask)
        score_miRACL = (scores_RACL[4] + (scores_MI[0]+scores_MI[2])/2) / 2
        end = time.time()

        metrics = {
            'opinion_f1': scores_RACL[0], 'OE_loss': losses[1],
            'aspect_f1': scores_RACL[1], 'AE_loss': losses[2],
            'sentiment_acc': scores_RACL[2], 'sentiment_f1': scores_RACL[3], 'SC_loss': losses[3],
            'ABSA_f1': scores_RACL[4], 'miRACL_f1': score_miRACL,
            'total_loss': losses[0], 'Reg_cost': losses[6],
            'category_f1': scores_MI[0], 'category_acc': scores_MI[1], 'CC_loss': losses[4], 
            'attribute_f1': scores_MI[2], 'attribute_acc': scores_MI[3], 'AC_loss': losses[5], 
        } 

        self.max_score_ABSA_index, \
        self.max_score_miRACL_index, self.min_loss_index = self.update_metrics(metrics)
        display_text = f'Epoch {epoch+1:03d} - Evaluation in {int(end-start)} seconds\n' + \
                       f'\t OE_loss={losses[1]:.3f}, AE_loss={losses[2]:.3f}, SC_loss={losses[3]:.3f}, CC_loss={losses[4]:.3f}, AC_loss={losses[5]:.3f}, Reg_cost={losses[6]:.3f}, total_loss={losses[0]:.3f}' + \
                       f'\n --> Best loss at Epoch {self.min_loss_index}' + \
                       f'\n\t opinion_f1={scores_RACL[0]:.7f}, aspect_f1={scores_RACL[1]:.7f}, sentiment_acc={scores_RACL[2]:.7f}, sentiment_f1={scores_RACL[3]:.7f}, ABSA_f1={scores_RACL[4]:.7f}' + \
                       f'\n --> Best ABSA-score at Epoch {self.max_score_ABSA_index}' + \
                       f'\n\t category_f1={scores_MI[0]:.7f}, category_acc={scores_MI[1]:.7f}, attribute_f1={scores_MI[2]:.7f}, attribute_acc={scores_MI[3]:.7f}, miRACL_f1={score_miRACL:.7f}' + \
                       f'\n --> Best miRACL-score at Epoch {self.max_score_miRACL_index}'
        self.logger.info(display_text)
        return metrics

    def update_metrics(self, metrics):
        for k, v in metrics.items():
            self.records[k].append(v)
        return np.argmax(self.records['ABSA_f1'])+1, \
               np.argmax(self.records['miRACL_f1'])+1, \
               np.argmin(self.records['total_loss'])+1


