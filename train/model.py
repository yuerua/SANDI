import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.layers import concatenate

def get_similarity_model(input_shape, loss_type):
    img_in = Input(shape = input_shape, name = 'FeatureNet_ImageInput')
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

    feature_model = Model(inputs = [img_in], outputs = [n_layer], name = 'FeatureGenerationModel')

    img_a_in = Input(shape = input_shape, name = 'ImageA_Input')
    img_b_in = Input(shape = input_shape, name = 'ImageB_Input')
    img_a_feat = feature_model(img_a_in)
    img_b_feat = feature_model(img_b_in)

    img_feat = Lambda(lambda x: tf.concat([x[0], x[1]], 1), name = "Contrast")([img_a_feat, img_b_feat])

    # img_a_feat = MLP()(img_a_feat)
    # img_b_feat = feature_model(img_b_in)
    combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')
    combined_features = Dense(16, activation = 'linear')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)
    combined_features = Dense(4, activation = 'linear')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)
    combined_features = Dense(1, activation = 'sigmoid', name = "BCE")(combined_features)

    if loss_type == "bce_only":
        outputs_ = [combined_features]
    elif loss_type == "contrast_only":
        outputs_ = [img_feat]
    else:
        outputs_ = [img_feat, combined_features]

    sim_model = Model(inputs = [img_a_in, img_b_in], outputs = outputs_, name = 'similarity')

    return sim_model

def MLP():
    inputs = Input((32,))
    x = Dense(32, activation = 'linear')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(32, activation = 'linear')(x)

    MLP = tf.keras.Model(inputs, x)

    return MLP

def similarity_branch():
    img_a_feat = Input((32,))
    img_b_feat = Input((32,))
    combined_features = concatenate([img_a_feat, img_b_feat], name = 'merge_features')
    combined_features = Dense(16, activation = 'linear')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)
    combined_features = Dense(4, activation = 'linear')(combined_features)
    combined_features = BatchNormalization()(combined_features)
    combined_features = Activation('relu')(combined_features)
    combined_features = Dense(1, activation = 'sigmoid')(combined_features)

    sim_model = Model(inputs = [img_a_feat, img_b_feat], outputs = [combined_features], name = 'Similarity')

    return sim_model



def get_supervised_classifier(input_shape, num_class):
    img_in = Input(shape = input_shape, name = 'FeatureNet_ImageInput')
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

    c_layer = Dense(16, activation = 'linear')(n_layer)
    c_layer = BatchNormalization()(c_layer)
    c_layer = Activation('relu')(c_layer)
    c_layer = Dense(16, activation = 'linear')(c_layer)
    c_layer = BatchNormalization()(c_layer)
    c_layer = Activation('relu')(c_layer)

    if num_class==2:
        print('Using sigmoid')
        c_layer = Dense(num_class, activation = 'sigmoid')(c_layer)
    else:
        print('Using softmax')
        c_layer = Dense(num_class, activation='softmax')(c_layer)

    classifier = Model(inputs = [img_in], outputs = [c_layer], name = 'Classifier')
    return classifier


