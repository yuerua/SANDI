import tensorflow as tf


def similar_sample(data_in, sub_patch_size):
    data = tf.map_fn(lambda img: tf.image.random_crop(img, (sub_patch_size, sub_patch_size, 3)), data_in)
    return data


def dissimilar_sample(data_in):

    data = tf.map_fn(lambda img: tf.image.random_hue(img, max_delta=0.5), data_in)
    data = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=0.5), data)
    data = tf.map_fn(lambda img: tf.image.random_contrast(img, lower=0.8, upper=1), data)
    data = tf.map_fn(lambda img: tf.image.random_saturation(img, lower=0.8, upper=1), data)

    return data

def augment(data_in):

    img_patch = tf.image.random_flip_left_right(data_in)
    img_patch = tf.image.random_flip_up_down(img_patch)

    return img_patch