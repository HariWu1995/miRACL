import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Layer, Embedding,
    Conv1D, Dropout, Dense,
    Concatenate, Lambda, )
from tensorflow.keras.models import Model, Sequential


##################################
#      Tensorflow Functions      #
##################################

def _bernoulli(shape, mean):
    return tf.nn.relu(
        tf.sign(mean-tf.random.uniform(shape, minval=0, maxval=1, dtype=tf.float32))
    )


def to_float(tensor):
    return tf.cast(tensor, tf.float32)


def gelu(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    refer : https://github.com/google-research/bert/blob/bee6030e31e42a9394ac567da170a89a98d2062f/modeling.py#L264
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.math.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.math.pow(x, 3)))))
    return x * cdf


##################################
#           Keras Layer          #
##################################

class DropBlock2D(Layer):

    # Adopted from https://github.com/DHZS/tf-dropblock

    def __init__(self, keep_prob: float=1.0, block_size: int=1, scale: bool=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = keep_prob
        self.block_size = block_size
        self.scale = tf.constant(scale, dtype=tf.bool)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size-1) // 2
        p0 = (self.block_size-1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.h-self.block_size+1,
                                        self.w-self.block_size+1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], padding='SAME')
        mask = 1 - mask
        return mask

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            mask_size = to_float(tf.size(mask))
            inputs_masked = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: inputs_masked*mask_size/tf.reduce_sum(mask),
                             false_fn=lambda: inputs_masked)
            return output

        if training is None:
            training = K.learning_phase()
            print(self.name, training)
        output = tf.cond(
            tf.logical_or(tf.logical_not(bool(training)),
                          tf.equal(self.keep_prob, 1.0)),
            true_fn=lambda: inputs,
            false_fn=drop
        )
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = to_float(self.w), to_float(self.h)
        dropout_rate = 1. - self.keep_prob
        self.gamma = dropout_rate*(w*h) / (self.block_size**2) / ((w-self.block_size+1)*(h-self.block_size+1))


class L2Norm(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True

    def call(self, inputs, mask=None):
        # return K.l2_normalize(inputs, axis=self.axis)
        return tf.math.l2_normalize(inputs, axis=self.axis)


class SoftMask2D(Layer):
    def __init__(self, scale: bool=False, **kwargs):
        super(SoftMask2D, self).__init__(**kwargs)
        self.scale = scale
        self.supports_masking = True

    def call(self, inputs):
        x, mask = inputs
        if self.scale:
            dim = tf.shape(x)[-1]
            max_x = tf.math.reduce_max(x, axis=-1, keepdims=True, name='max_x')
            max_x = tf.tile(max_x, [1, 1, dim], name='max_x_tiled')
            x = tf.math.subtract(x, max_x, name='x_scaled')
        length = tf.shape(mask)[1]
        mask_d1 = tf.tile(tf.expand_dims(mask, axis=1), [1, length, 1], name='mask_d1')
        y = tf.math.multiply(tf.exp(x), mask_d1, name='y')
        sum_y = tf.math.reduce_sum(y, axis=-1, keepdims=True, name='sum_y')
        att = tf.math.divide(y, sum_y+K.epsilon(), name='att')

        mask_d2 = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, length], name='mask_d2')
        att = tf.math.multiply(att, mask_d2, name='att_masked')
        return att


class ExpandDim(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ExpandDim, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = False

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


class Squeeze(Layer):
    def __init__(self, axis=-1, **kwargs):
        super(Squeeze, self).__init__(**kwargs)
        self.axis = axis
        self.supports_masking = False

    def call(self, inputs):
        return tf.squeeze(inputs, axis=self.axis)


class ReduceDim(Layer):
    def __init__(self, method: str='mean', axis=None, **kwargs):
        super(ReduceDim, self).__init__(**kwargs)
        self.axis = axis
        self.method = method.lower()
        self.supports_masking = False

    def call(self, inputs):
        if self.method == 'sum':
            return tf.math.reduce_sum(inputs, axis=self.axis)
        elif self.method == 'mean':
            return tf.math.reduce_mean(inputs, axis=self.axis)
        elif self.method == 'max':
            return tf.math.reduce_max(inputs, axis=self.axis)
        elif self.method == 'min':
            return tf.math.reduce_min(inputs, axis=self.axis)
        elif self.method == 'std':
            return tf.math.reduce_std(inputs, axis=self.axis)
        elif self.method == 'variance':
            return tf.math.reduce_variance(inputs, axis=self.axis)
        else:
            raise ValueError(f'method={self.method} has been implemented yet!')


class MatMul(Layer):
    def __init__(self, adjoint_a=False, adjoint_b=False,
                    transpose_a=False, transpose_b=False,
                    a_is_sparse=False, b_is_sparse=False, **kwargs):
        super(MatMul, self).__init__(**kwargs)
        self.adjoint_a, self.adjoint_b = adjoint_a, adjoint_b
        self.transpose_a, self.transpose_b = transpose_a, transpose_b
        self.a_is_sparse, self.b_is_sparse = a_is_sparse, b_is_sparse
        self.supports_masking = False

    def call(self, inputs):
        args = {
            'a': inputs[0], 'b': inputs[1],
            'adjoint_a': self.adjoint_a, 'adjoint_b': self.adjoint_b,
            'transpose_a': self.transpose_a, 'transpose_b': self.transpose_b,
            'a_is_sparse': self.a_is_sparse, 'b_is_sparse': self.b_is_sparse,
        }
        return tf.linalg.matmul(**args)


class RACL_Block(Model):

    def __init__(self, opt, Normalizer, DropBlocks, TileBlock,
                conv_args: dict, classifier_args: dict, block_id: int, **kwargs):
        super(RACL_Block, self).__init__(**kwargs)
        self._name = f'RACL_Block_{block_id}'
        self.opt = opt
        self.block_id = block_id

        self.Tile = TileBlock
        self.Normalizer = Normalizer
        self.DropBlock_aspect, self.DropBlock_opinion, self.DropBlock_context = DropBlocks

        self.Aspect_Extractor = Conv1D(filters=self.opt.n_filters, name=f'Aspect_Conv-{block_id}', **conv_args)
        self.Opinion_Extractor = Conv1D(filters=self.opt.n_filters, name=f'Opinion_Conv-{block_id}', **conv_args)
        self.Context_Extractor = Conv1D(filters=self.opt.embedding_dim, name=f'Context_Conv-{block_id}', **conv_args)

        self.Aspect_Classifier = Dense(name=f'Aspect_Classifier-{block_id}', **classifier_args['aspect'])
        self.Opinion_Classifier = Dense(name=f'Opinion_Classifier-{block_id}', **classifier_args['opinion'])
        self.Sentiment_Classifier = Dense(name=f'Sentiment_Classifier-{block_id}', **classifier_args['sentiment'])
        self.Category_Classifier = Dense(name=f'Category_Classifier-{block_id}', **classifier_args['category'])
        self.Attribute_Classifier = Dense(name=f'Attribute_Classifier-{block_id}', **classifier_args['attribute'])

    def call(self, inputs):
        aspect_input, opinion_input, context_input, context_query, word_mask, position_att = inputs
        i = self.block_id

        # Extract Private Features for each task
        aspect_conv = self.Aspect_Extractor(aspect_input)
        opinion_conv = self.Opinion_Extractor(opinion_input)
        context_conv = self.Context_Extractor(context_input)

        # Normalize
        aspect_conv_norm = self.Normalizer(aspect_conv)
        opinion_conv_norm = self.Normalizer(opinion_conv)
        context_conv_norm = self.Normalizer(context_conv)

        # Relation R1
        aspect_see_opinion = MatMul(adjoint_b=True, name=f'aspect_see_opinion-{i}')([aspect_conv_norm, opinion_conv_norm])
        aspect_attend_opinion = SoftMask2D(name=f'aspect_attend_opinion-{i}')([aspect_see_opinion, word_mask])
        aspect_weigh_opinion = MatMul(name=f'aspect_weigh_opinion-{i}')([aspect_attend_opinion, opinion_conv])
        aspect_interact = Concatenate(axis=-1, name=f'aspect_interact-{i}')([aspect_conv, aspect_weigh_opinion])

        opinion_see_aspect = MatMul(adjoint_b=True, name=f'opinion_see_aspect-{i}')([opinion_conv_norm, aspect_conv_norm])
        opinion_attend_aspect = SoftMask2D(name=f'opinion_attend_aspect-{i}')([opinion_see_aspect, word_mask])
        opinion_weigh_aspect = MatMul(name=f'opinion_weigh_aspect-{i}')([opinion_attend_aspect, aspect_conv])
        opinion_interact = Concatenate(axis=-1, name=f'opinion_interact-{i}')([opinion_conv, opinion_weigh_aspect])

        # AE & OE Prediction + Multi-Label (2-level) Prediction
        aspect_pred = self.Aspect_Classifier(aspect_interact)
        opinion_pred = self.Opinion_Classifier(opinion_interact)
        category_pred = self.Category_Classifier(aspect_interact)
        attribute_pred = self.Attribute_Classifier(aspect_interact)

        # OE Confidence - a slight difference from the original paper.
        # For propagating R3, we calculate the confidence of each candidate opinion word.
        # Only when a word satisfies the condition Prob[B,I] > Prob[O] in OE, it can be propagated to SC.
        opinion_condition = Lambda(lambda x: 1-2.*tf.nn.softmax(x, axis=-1)[:,:,0], name=f'opinion_condition-{i}')(opinion_pred)
        opinion_confidence = Lambda(lambda x: tf.math.maximum(0., x), name=f'opinion_confidence-{i}')(opinion_condition)
        mask = self.Tile(word_mask)
        opinion_propagated = self.Tile(opinion_confidence)
        opinion_propagated = MatMul(name=f'opinion_propagated_masked-{i}')([opinion_propagated, mask])
        opinion_propagated = MatMul(name=f'opinion_propagated-{i}')([opinion_propagated, position_att])

        # SC Aspect-Context Attention
        word_see_context = MatMul(adjoint_b=True, name=f'word_see_context-{i}')([(context_query), context_conv_norm])
        word_see_context = MatMul(name=f'word_see_context_masked-{i}')([word_see_context, position_att])
        word_attend_context = SoftMask2D(scale=True, name=f'word_attend_context-{i}')([word_see_context, word_mask])

        # Relation R2 & R3
        word_attend_context += aspect_attend_opinion + opinion_propagated
        word_weigh_context = MatMul(name=f'word_weigh_context-{i}')([word_attend_context, context_conv])
        context_interact = context_query + word_weigh_context

        # SC Prediction
        sentiment_pred = self.Sentiment_Classifier(context_interact)

        # We use DropBlock to enhance the learning of the private features for AE, OE & SC.
        # For more details, refer to
        #   http://papers.nips.cc/paper/8271-dropblock-a-regularization-method-for-convolutional-networks for more details.
        aspect_interact = ExpandDim(axis=-1)(aspect_interact)
        aspect_interact = self.DropBlock_aspect(aspect_interact, self.opt.is_training)
        aspect_interact = Squeeze(axis=-1)(aspect_interact)

        opinion_interact = ExpandDim(axis=-1)(opinion_interact)
        opinion_interact = self.DropBlock_opinion(opinion_interact, self.opt.is_training)
        opinion_interact = Squeeze(axis=-1)(opinion_interact)

        context_conv = ExpandDim(axis=-1)(context_conv)
        context_conv = self.DropBlock_context(context_conv, self.opt.is_training)
        context_conv = Squeeze(axis=-1)(context_conv)

        return [(aspect_pred, opinion_pred, sentiment_pred, category_pred, attribute_pred),
                (aspect_interact, opinion_interact, context_interact, context_conv)]


def dropoutize_embeddings(opt, layer_name: str='embeddings_dropout', model_name: str='dropoutize_embeddings'):
    model = Sequential(name=model_name)
    model.add(Dropout(rate=1-opt.keep_prob_1,
                      input_shape=(opt.max_sentence_len, opt.embedding_dim),
                      seed=opt.random_seed,
                      name=layer_name))
    word_inputs, word_embeddings = model.inputs, model.outputs
    return word_inputs[0], word_embeddings[0]


def create_embeddings(inputs, opt, embedding_dim: int, layer_prefix: str='', pretrained_embeddings=None):
    embedding_args = {
        'input_dim': opt.vocab_size+1, # MUST: +1
        'output_dim': embedding_dim,
        'name': layer_prefix+'_embeddings',
        'trainable': False # finetune only happens after warm-up
    }
    if pretrained_embeddings is not None \
        and pretrained_embeddings.shape==(embedding_args['input_dim'], embedding_args['output_dim']):
        embedding_args['weights'] = [pretrained_embeddings]
    embeddings = Embedding(**embedding_args)(inputs)
    embeddings = Dropout(1-opt.keep_prob_1, name=layer_prefix+'_embeddings_dropout')(embeddings)
    return embeddings


if __name__ == "__main__":
    x = tf.keras.Input(shape=(12, 10, 100))
    y = DropBlock2D(0.5, 1, name='DropBlock2D')(x)
    model = tf.keras.Model(inputs=[x], outputs=[y], name='test')
    model.summary()











