import tensorflow as tf
import tensorflow.keras.backend as K


def weighted_cross_entropy(y_true, y_pred, pos_weight=7):
    losses = y_true * -K.log(y_pred) * pos_weight + (1-y_true) * -K.log(1-y_pred)
    losses = K.clip(losses, 0.0, 11)
    return K.mean(losses)


def balanced_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        background_weights = weights[:,0] ** (1-y_true)
        signal_weights = weights[:,1] ** (y_true)
        CE = K.binary_crossentropy(y_true, y_pred)
        losses = background_weights * signal_weights * CE
        losses = K.clip(losses, 0.0, 11.0)
        return K.mean(losses, axis=-1)
    return weighted_loss


def crossentropy(preds, labels, weights=None, name='loss'):

    with tf.name_scope(name) as scope:
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels, name='unweighted_loss')

        if weights is None:
            loss = tf.reduce_mean(loss, dtype=tf.float32, name='final_loss')
            return loss

        class_weights = tf.constant(weights, dtype=tf.float32, name='class_weights')
        try:
            weighted_samples = tf.math.multiply(class_weights, labels, name='weighted_samples_pre')
        except tf.errors.InvalidArgumentError as e:
            print(class_weights)
            print(labels)
            exit()
        weighted_samples = tf.reduce_sum(weighted_samples, axis=1, name='weighted_samples')
        loss = tf.reduce_mean(weighted_samples*loss, name='weighted_loss')
        return loss


def RACL_losses(y_true, y_pred, masks, opt):
    ae_label, oe_label, sc_label, cc_label, ac_label = y_true
    ae_pred , oe_pred , sc_pred , cc_pred , ac_pred  = y_pred
    source_mask, sentiment_mask = masks

    # Format predictions
    ae_pred = tf.cast(ae_pred, tf.float32, name='ae_pred')
    oe_pred = tf.cast(oe_pred, tf.float32, name='oe_pred')
    sc_pred = tf.cast(sc_pred, tf.float32, name='sc_pred')
    cc_pred = tf.cast(cc_pred, tf.float32, name='cc_pred')
    ac_pred = tf.cast(ac_pred, tf.float32, name='ac_pred')
    source_mask = tf.cast(source_mask, tf.float32, name='source_mask')
    sentiment_mask = tf.cast(sentiment_mask, tf.float32, name='sentiment_mask')

    # Convert values to probabilities
    ae_prob = tf.nn.softmax(ae_pred, axis=-1, name='ae_prob')
    oe_prob = tf.nn.softmax(oe_pred, axis=-1, name='oe_prob')
    # sc_prob = tf.nn.softmax(sc_pred, axis=-1, name='sc_prob')

    # Mask AE, OE, SC Predictions
    aeoe_mask = tf.tile(tf.expand_dims(source_mask, axis=-1), [1,1,3])
    ae_pred = tf.reshape(aeoe_mask*ae_pred, [-1,3], name='ae_pred_masked')
    oe_pred = tf.reshape(aeoe_mask*oe_pred, [-1,3], name='oe_pred_masked')
    
    cc_mask = tf.tile(tf.expand_dims(source_mask, axis=-1), [1, 1, opt.n_categories])
    cc_pred = tf.reshape(cc_mask*cc_pred, [-1, opt.n_categories], name='cc_pred_masked')

    ac_mask = tf.tile(tf.expand_dims(source_mask, axis=-1), [1, 1, opt.n_attributes])
    ac_pred = tf.reshape(ac_mask*ac_pred, [-1, opt.n_attributes], name='ac_pred_masked')

    sentiment_mask = tf.tile(tf.expand_dims(sentiment_mask, axis=-1), [1, 1, opt.n_sentiments])
    sc_pred = tf.reshape(sentiment_mask*sc_pred, [-1, opt.n_sentiments], name='sc_pred_masked')

    # Relation R4 (only in Training)
    # In training / validation, sentiment masks are set to 1.0 only for aspect terms.
    # In testing, sentiment masks are set to 1.0 for all words (except padded ones).

    # Format Labels
    ae_label = tf.cast(ae_label, tf.float32, name='ae_label')
    oe_label = tf.cast(oe_label, tf.float32, name='oe_label')
    sc_label = tf.cast(sc_label, tf.float32, name='sc_label')
    cc_label = tf.cast(cc_label, tf.float32, name='cc_label')
    ac_label = tf.cast(ac_label, tf.float32, name='ac_label')
    ae_label = tf.reshape(ae_label, [-1,3], name='ae_label_flat')
    oe_label = tf.reshape(oe_label, [-1,3], name='oe_label_flat')
    sc_label = tf.reshape(sc_label, [-1, opt.n_sentiments], name='sc_label_flat')
    cc_label = tf.reshape(cc_label, [-1, opt.n_categories], name='cc_label_flat')
    ac_label = tf.reshape(ac_label, [-1, opt.n_attributes], name='ac_label_flat')

    # AE & OE Regularization cost - only get Beginning [1] and Inside [2] values
    ae_cost = tf.reduce_sum(ae_prob[:,:,1:], axis=-1, name='ae_cost')
    oe_cost = tf.reduce_sum(oe_prob[:,:,1:], axis=-1, name='oe_cost')
    total_cost = ae_cost + oe_cost - 1.
    total_cost = tf.maximum(0., total_cost, name='total_cost')
    reg_cost = tf.reduce_sum(total_cost) / tf.reduce_sum(aeoe_mask)
    reg_cost = tf.identity(reg_cost, name='regularization_cost')

    # Weighted SoftMax Categorical Cross-Entropy for AE, OE, SC
    ae_loss = crossentropy(ae_pred, ae_label, opt.term_weights, name='aspect')
    oe_loss = crossentropy(oe_pred, oe_label, opt.term_weights, name='opinion')
    sc_loss = crossentropy(sc_pred, sc_label, opt.polarity_weights, name='sentiment')
    cc_loss = crossentropy(cc_pred, cc_label, opt.category_weights, name='category')
    ac_loss = crossentropy(ac_pred, ac_label, opt.attribute_weights, name='attribute')

    loss = opt.ae_weight * ae_loss + \
           opt.oe_weight * oe_loss + \
           opt.sc_weight * sc_loss + \
           opt.cc_weight * cc_loss + \
           opt.ac_weight * ac_loss + \
           opt.reg_weight * reg_cost
    loss = tf.identity(loss, name='overall_loss')
    return loss, ae_loss, oe_loss, sc_loss, cc_loss, ac_loss, reg_cost


def RACL_losses_wrapper(opt):
    def call_RACL_losses(y_true, y_pred):
        # Split predictions into 
        #       source_mask, sentiment_mask, 
        #       aspect_pred, opinion_pred, sentiment_pred, category_pred, attribute_pred
        source_mask, sentiment_mask, *predictions = tf.split(y_pred, [1, 1, 3, 3, opt.n_sentiments, opt.n_categories, opt.n_attributes], axis=-1)
        source_mask = tf.squeeze(source_mask, axis=-1)
        sentiment_mask = tf.squeeze(sentiment_mask, axis=-1)

        # Split labels into aspect_label, opinion_label, sentiment_label, category_label, attribute_label
        labels = tf.split(y_true, [3, 3, opt.n_sentiments, opt.n_categories, opt.n_attributes], axis=-1)

        # Call RACL_losses
        losses = RACL_losses(predictions, labels, [source_mask, sentiment_mask], opt)
        return losses[0]
    return call_RACL_losses



