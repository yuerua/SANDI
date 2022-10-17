import tensorflow as tf
from tensorflow.keras import backend as K

#loss
class Loss:

    def __init__(self, opts):
        self.weight1 = opts.weight1
        self.weight0 = opts.weight0
        self.batch_size = opts.batch_size
        self.loss_temp = opts.loss_temp
        self.loss_type = opts.loss_type
        self.supervised = opts.supervised

    def loss_function(self):
        # loss
        if self.loss_type == "bce_only":
            print("Using bce loss")
            loss_ = self.wbce
            if self.supervised:
                print("Supervised training")
            else:
                print("Unsupervised training")

        elif self.loss_type == "contrast_only":
            print("Using contrastive loss")
            if self.supervised:
                print("Supervised training")
                loss_ = self.contrast_loss_supervised
            else:
                print("Unsupervised training")
                loss_ = self.contrast_loss

        else:
            print("Using combined loss")
            if self.supervised:
                print("Supervised training")
                loss_ = [self.contrast_loss_supervised, self.wbce]
            else:
                print("Unsupervised training")
                loss_ = [self.contrast_loss, self.wbce]

        return loss_


    def wbce(self, y_true, y_pred):
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        logloss = -(y_true * K.log(y_pred) * self.weight1 + (1 - y_true) * K.log(1 - y_pred) * self.weight0)

        return K.mean(logloss, axis=-1)

    def contrast_loss(self, y_true, emb):
        LARGE_NUM = 1e9

        img_a_feat, img_b_feat = tf.split(emb, 2, 1)

        img_a_feat = tf.nn.l2_normalize(img_a_feat, axis=-1)
        img_b_feat = tf.nn.l2_normalize(img_b_feat, axis=-1)

        hidden1, _ = tf.split(img_a_feat, 2, 0)
        hidden2, _ = tf.split(img_b_feat, 2, 0)

        hidden1_large = hidden1
        hidden2_large = hidden2

        labels = tf.one_hot(tf.range(self.batch_size), self.batch_size * 2)
        masks = tf.one_hot(tf.range(self.batch_size), self.batch_size)

        # SimCLR contrast loss
        logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / self.loss_temp
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / self.loss_temp
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / self.loss_temp
        logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / self.loss_temp

        loss_a = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = tf.losses.softmax_cross_entropy(
            labels, tf.concat([logits_ba, logits_bb], 1))
        contrast_loss = loss_a + loss_b

        return contrast_loss

    def contrast_loss_supervised(self, y_true, emb):
        num_views = 2
        num_anchor_views = 2

        img_a_feat, img_b_feat = tf.split(emb, 2, 1)

        img_a_feat = tf.nn.l2_normalize(img_a_feat, axis=-1)
        img_b_feat = tf.nn.l2_normalize(img_b_feat, axis=-1)

        hidden_1, _ = tf.split(img_a_feat, 2, 0)
        hidden_2, _ = tf.split(img_b_feat, 2, 0)

        features = tf.concat([tf.expand_dims(hidden_1, 1), tf.expand_dims(hidden_2, 1)], 1)
        global_features = features

        labels_true, _ = tf.split(y_true, 2, 0)

        all_global_features = tf.reshape(
            tf.transpose(global_features, perm=[1, 0, 2]),
            [num_views * self.batch_size, -1])

        anchor_features = tf.reshape(
            tf.transpose(features, perm=[1, 0, 2]),
            [num_views * self.batch_size, -1])

        mask = tf.matmul(labels_true, labels_true, transpose_b=True)

        # identity matrix to indicate the position of positive pairs from the same image
        diagonal_mask = tf.one_hot(tf.range(self.batch_size), self.batch_size)
        # all_but_diagonal_mask = 1. - diagonal_mask
        # negatives_mask = 1. - mask
        # positives_mask = mask #* all_but_diagonal_mask
        labels = tf.argmax(diagonal_mask, axis=-1)
        tiled_labels = []
        for i in range(num_anchor_views):
            tiled_labels.append(labels + tf.cast(self.batch_size, labels.dtype) * i)
        tiled_labels = tf.concat(tiled_labels, axis=0)
        tiled_diagonal_mask = tf.one_hot(tiled_labels, self.batch_size * num_views)
        all_but_diagonal_mask = 1. - tiled_diagonal_mask

        uncapped_positives_mask = tf.tile(mask, [num_anchor_views, num_views])
        negatives_mask = 1. - uncapped_positives_mask
        positives_mask = uncapped_positives_mask * all_but_diagonal_mask

        logits = tf.matmul(
            anchor_features, all_global_features, transpose_b=True)
        temperature = tf.cast(self.loss_temp, tf.float32)
        logits = logits / temperature
        logits = (
                logits - tf.reduce_max(tf.stop_gradient(logits), axis=1, keepdims=True))
        exp_logits = tf.exp(logits)

        # The following masks are all tiled by the number of views, i.e., they have
        # shape [local_batch_size * num_anchor_views, global_batch_size * num_views].
        num_positives_per_row = tf.reduce_sum(positives_mask, axis=1)

        denominator = tf.reduce_sum(
            exp_logits * negatives_mask, axis=1, keepdims=True) + tf.reduce_sum(
            exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = (logits - tf.log(denominator)) * positives_mask
        log_probs = tf.reduce_sum(log_probs, axis=1)

        log_probs = tf.divide(log_probs, num_positives_per_row)
        log_probs = tf.where(tf.is_nan(log_probs), tf.zeros_like(log_probs), log_probs)

        contrast_loss = -log_probs

        contrast_loss = tf.reduce_mean(contrast_loss, axis=0)

        return contrast_loss






