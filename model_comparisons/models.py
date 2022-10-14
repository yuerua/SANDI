#Modified from https://github.com/beresandras/contrastive-classification-keras/

import tensorflow as tf

from abc import abstractmethod
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout


class ContrastiveModel(keras.Model):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
    ):
        super().__init__()

        self.contrastive_augmenter = contrastive_augmenter
        self.classification_augmenter = classification_augmenter
        self.encoder = encoder
        self.projection_head = projection_head
        self.linear_probe = linear_probe

    def compile(self, contrastive_optimizer, probe_optimizer, **kwargs):
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss will be defined as a method
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy()
        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()
        self.probe_accuracy.reset_states()

    def update_contrastive_accuracy(self, features_1, features_2):
        # self-supervised metric inspired by the SimCLR loss

        # cosine similarity: the dot product of the l2-normalized feature vectors
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        # the similarity between the representations of two augmented views of the
        # same image should be higher than their similarity with other views
        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        # self-supervised metric inspired by the BarlowTwins loss

        # normalization so that cross-correlation will be between -1 and 1
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        # the cross correlation of image representations should be the identity matrix
        # batch_size = tf.shape(features_1, out_type=tf.float32)[0]
        batch_size = tf.shape(features_1)[0]
        batch_size = tf.cast(batch_size, tf.float32)
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2):
        pass

    def train_step(self, data):
        # (unlabeled_images, _), (labeled_images, labels) = data
        (unlabeled_images, _) = data

        # both labeled and unlabeled images are used, without labels
        # images = tf.concat((unlabeled_images, labeled_images), axis=0)
        images = unlabeled_images
        # each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            # the representations are passed through a projection mlp
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        # labels are only used in evalutation for an on-the-fly logistic regression
        # preprocessed_images = self.classification_augmenter(labeled_images)
        # with tf.GradientTape() as tape:
        #     features = self.encoder(preprocessed_images)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result()
            # "p_loss": probe_loss,
            # "p_acc": self.probe_accuracy.result(),
        }

    def test_step(self, data):
        # labeled_images, labels = data
        #
        # preprocessed_images = self.classification_augmenter(
        #     labeled_images, training=False
        # )
        # features = self.encoder(preprocessed_images, training=False)
        # class_logits = self.linear_probe(features, training=False)
        # probe_loss = self.probe_loss(labels, class_logits)
        #
        # self.probe_accuracy.update_state(labels, class_logits)
        # return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()}
        # (unlabeled_images, _), (labeled_images, labels) = data
        (unlabeled_images, _) = data

        # both labeled and unlabeled images are used, without labels
        # images = tf.concat((unlabeled_images, labeled_images), axis=0)
        images = unlabeled_images
        # each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images,  training=False)
        augmented_images_2 = self.contrastive_augmenter(images,  training=False)
        # with tf.GradientTape() as tape:
        features_1 = self.encoder(augmented_images_1,  training=False)
        features_2 = self.encoder(augmented_images_2,  training=False)
        # the representations are passed through a projection mlp
        projections_1 = self.projection_head(features_1,  training=False)
        projections_2 = self.projection_head(features_2,  training=False)
        contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        contrastive_loss = tf.reduce_mean(contrastive_loss)
        # gradients = tape.gradient(
        #     contrastive_loss,
        #     self.encoder.trainable_weights + self.projection_head.trainable_weights,
        # )
        # self.contrastive_optimizer.apply_gradients(
        #     zip(
        #         gradients,
        #         self.encoder.trainable_weights + self.projection_head.trainable_weights,
        #     )
        # )
        # self.update_contrastive_accuracy(features_1, features_2)
        # self.update_correlation_accuracy(features_1, features_2)

        # labels are only used in evalutation for an on-the-fly logistic regression
        # preprocessed_images = self.classification_augmenter(labeled_images)
        # with tf.GradientTape() as tape:
        #     features = self.encoder(preprocessed_images)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        # return {
        #     "c_loss": contrastive_loss,
        #     "c_acc": self.contrastive_accuracy.result(),
        #     "r_acc": self.correlation_accuracy.result()
        #     "p_loss": probe_loss,
        #     "p_acc": self.probe_accuracy.result(),
        # }
        return {"c_loss":contrastive_loss}


class MomentumContrastiveModel(ContrastiveModel):
    def __init__(
        self,
        contrastive_augmenter,
        classification_augmenter,
        encoder,
        projection_head,
        linear_probe,
        momentum_coeff,
    ):
        super().__init__(
            contrastive_augmenter,
            classification_augmenter,
            encoder,
            projection_head,
            linear_probe,
        )
        self.momentum_coeff = momentum_coeff

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)
        self.m_projection_head = keras.models.clone_model(self.projection_head)

    @abstractmethod
    def contrastive_loss(
        self,
        projections_1,
        projections_2,
        m_projections_1,
        m_projections_2,
    ):
        pass

    def train_step(self, data):
        # (unlabeled_images, _), (labeled_images, labels) = data

        # images = tf.concat((unlabeled_images, labeled_images), axis=0)
        (unlabeled_images, _) = data
        images = unlabeled_images

        augmented_images_1 = self.contrastive_augmenter(images)
        augmented_images_2 = self.contrastive_augmenter(images)
        with tf.GradientTape() as tape:
            features_1 = self.encoder(augmented_images_1)
            features_2 = self.encoder(augmented_images_2)
            projections_1 = self.projection_head(features_1)
            projections_2 = self.projection_head(features_2)
            m_features_1 = self.m_encoder(augmented_images_1)
            m_features_2 = self.m_encoder(augmented_images_2)
            m_projections_1 = self.m_projection_head(m_features_1)
            m_projections_2 = self.m_projection_head(m_features_2)
            contrastive_loss = self.contrastive_loss(
                projections_1, projections_2, m_projections_1, m_projections_2
            )
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projection_head.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projection_head.trainable_weights,
            )
        )
        self.update_contrastive_accuracy(m_features_1, m_features_2)
        self.update_correlation_accuracy(m_features_1, m_features_2)

        # preprocessed_images = self.classification_augmenter(labeled_images)
        # with tf.GradientTape() as tape:
        #     # the momentum encoder is used here as it moves more slowly
        #     features = self.m_encoder(preprocessed_images)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        # the momentum networks are updated by exponential moving average
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(
                self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight
            )
        for weight, m_weight in zip(
            self.projection_head.weights, self.m_projection_head.weights
        ):
            m_weight.assign(
                self.momentum_coeff * m_weight + (1 - self.momentum_coeff) * weight
            )

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result()
            # "p_loss": probe_loss,
            # "p_acc": self.probe_accuracy.result(),
        }

    def test_step(self, data):
        (unlabeled_images, _) = data

        # both labeled and unlabeled images are used, without labels
        # images = tf.concat((unlabeled_images, labeled_images), axis=0)
        images = unlabeled_images
        # each image is augmented twice, differently
        augmented_images_1 = self.contrastive_augmenter(images,  training=False)
        augmented_images_2 = self.contrastive_augmenter(images,  training=False)
        # with tf.GradientTape() as tape:
        features_1 = self.encoder(augmented_images_1,  training=False)
        features_2 = self.encoder(augmented_images_2,  training=False)
        projections_1 = self.projection_head(features_1,  training=False)
        projections_2 = self.projection_head(features_2,  training=False)
        m_features_1 = self.m_encoder(augmented_images_1,  training=False)
        m_features_2 = self.m_encoder(augmented_images_2,  training=False)
        m_projections_1 = self.m_projection_head(m_features_1,  training=False)
        m_projections_2 = self.m_projection_head(m_features_2,  training=False)
        contrastive_loss = self.contrastive_loss(
            projections_1, projections_2, m_projections_1, m_projections_2
        )
        contrastive_loss = tf.reduce_mean(contrastive_loss)

        return {"c_loss": contrastive_loss}


def get_feature_encoder(input_shape):
    img_in = Input(shape = input_shape, name = 'featurenet_imageinput')
    n_layer = img_in
    for i in range(2):
        n_layer = Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear')(n_layer)
        n_layer = BatchNormalization()(n_layer)
        n_layer = Activation('relu')(n_layer)
        n_layer = MaxPool2D((2,2))(n_layer)
    n_layer = Flatten()(n_layer)
    n_layer = Dense(32, activation = 'linear')(n_layer)
    n_layer = Dropout(0.5)(n_layer)
    n_layer = BatchNormalization()(n_layer)
    n_layer = Activation('relu')(n_layer)

    feature_encoder = Model(inputs = [img_in], outputs = [n_layer], name = 'encoder')
    feature_encoder.summary()
    return feature_encoder

def get_feature_encoder_sequential(input_shape):
    feature_encoder = keras.Sequential()
    feature_encoder.add(Input(shape = input_shape, name = 'featurenet_imageinput'))

    for i in range(2):
        feature_encoder.add(Conv2D(8*2**i, kernel_size = (3,3), activation = 'linear'))
        feature_encoder.add(BatchNormalization())
        feature_encoder.add(Activation('relu'))
        feature_encoder.add(Conv2D(16*2**i, kernel_size = (3,3), activation = 'linear'))
        feature_encoder.add(BatchNormalization())
        feature_encoder.add(Activation('relu'))
        feature_encoder.add(MaxPool2D((2,2)))
    feature_encoder.add(Flatten())
    feature_encoder.add(Dense(32, activation = 'linear'))
    feature_encoder.add(Dropout(0.5))
    feature_encoder.add(BatchNormalization())
    feature_encoder.add(Activation('relu'))

    # feature_encoder = Model(inputs = [img_in], outputs = [n_layer], name = 'encoder')
    feature_encoder.summary()
    return feature_encoder