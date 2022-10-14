import tensorflow as tf

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    # if isinstance(value, type(tf.constant(0))):
    #     value = value.numpy()
    if type(value) == str:
        value = value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode(in_feat, labels):
    # if len(labels.shape) == 2:
    #     labels = np.expand_dims(labels, axis=2)

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'in_feat/shape': int64_list_feature(in_feat.shape),
                'in_feat/data': bytes_feature(in_feat.tostring()),
                'labels/shape': int64_list_feature(labels.shape),
                'labels/data': bytes_feature(labels.tostring())}))

    return tf_example.SerializeToString()


def decode(serialized_example):
    # To read the names of features:  head -n10 /path/to/tfrecords
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'in_feat/shape': tf.io.FixedLenFeature([3], tf.int64),  # [3]: 3rows; shape of data, eg. (224,224,3)
            'in_feat/data': tf.io.FixedLenFeature([], tf.string),
            'labels/shape': tf.io.FixedLenFeature([1], tf.int64),
            'labels/data': tf.io.FixedLenFeature([], tf.string)
        })

    in_feat = tf.io.decode_raw(features['in_feat/data'], tf.uint8)  # Change this to uint16 or unit8
    # Change img to int32, with shape matching the input shape
    in_feat = tf.reshape(in_feat, tf.cast(features['in_feat/shape'], tf.int32))

    # Normalization
    # in_feat = tf.divide(tf.cast(in_feat, tf.float32), 255.0)  # 65535 #normalize
    in_feat = tf.cast(in_feat, tf.uint8)

    labels = tf.io.decode_raw(features['labels/data'], tf.uint8)
    labels = tf.cast(labels, tf.int64)
    return in_feat, labels