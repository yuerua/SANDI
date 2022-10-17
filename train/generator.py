# Description: generate pairs of patches with pair-wise label

import os
import tensorflow as tf
# from tensorflow.keras import backend as K
from util.data_utils import decode
from util.data_transform import similar_sample, augment

tf.compat.v1.disable_eager_execution()

class generator(object):
	def __init__(self, opts):

		"""Initialization:this initializes all parameters needed by the generator"""

		self.patch_size = opts.patch_size
		self.sub_patch_size = opts.sub_patch_size
		self.batch_size = opts.batch_size
		self.supervised = opts.supervised
		self.loss_type = opts.loss_type

		self.data_dir = opts.data_path
		self.train_data_filename = opts.train_data_filename
		self.valid_data_filename = opts.valid_data_filename

		print(self.data_dir)
		assert os.path.isfile(os.path.join(self.data_dir, self.train_data_filename + '.tfrecords')), \
			"Training tfrecords not found"
		assert os.path.isfile(os.path.join(self.data_dir, self.valid_data_filename + '.tfrecords')), \
			"Training tfrecords not found"

		assert self.loss_type in ["bce_only", "contrast_only", "combined"], \
			"Unrecognized loss type, using combined loss as default"


	def generate(self, type_):
		'Get batches'
		dataset = self.read_tfrecord(type_)

		data, label_class = dataset
		data_anchor = similar_sample(data, self.sub_patch_size)
		data_1 = similar_sample(data, self.sub_patch_size)

		data_0 = tf.reverse(data_1, [0])
		label_0 = tf.reverse(label_class, [0])

		data_a = tf.concat([data_anchor, data_anchor], axis=0)
		data_b = tf.concat([data_1, data_0], axis=0)

		pos_label = tf.constant([1.] * self.batch_size, dtype='float32')

		if self.supervised:
			#print("Using supervised labels")
			neg_label = tf.diag_part(tf.matmul(label_class, label_0, transpose_b=True))
			label_class_merge = tf.concat([label_class, label_0], 0)
		else:
			#print("Training without ground truth labels")
			neg_label = tf.constant([0.] * self.batch_size, dtype='float32')
			label_class_merge = tf.concat([pos_label, neg_label], 0)

		label = tf.concat([pos_label, neg_label], 0)

		if self.loss_type == "contrast_only":
			#print("Using contrastive loss")
			output = label_class_merge
		elif self.loss_type == "bce_only":
			#print("Using bce loss")
			output = label
		else:
			#print("Using combined loss")
			output = [label_class_merge, label]

		while True:
			yield tf.compat.v1.keras.backend.get_session().run(([data_a, data_b], output))
			#print(label_class_merge.eval(session = K.get_session()))

	def generate_subset(self, type_, subset_size):
		'Get batches'
		dataset = self.read_tfrecord_subset(type_, subset_size)

		data, label_class = dataset

		if type_ == "t":
			data = augment(data)
		while True:
			yield tf.compat.v1.keras.backend.get_session().run((data, label_class))

	def get_data_set(self, filename, shuffle_size, batch_size, prefetch_buffer):
		data_set = tf.data.TFRecordDataset(filename)
		data_set = data_set.map(decode)
		data_set = data_set.prefetch(prefetch_buffer)
		data_set = data_set.repeat()
		data_set = data_set.shuffle(shuffle_size)
		data_set = data_set.batch(batch_size)
		return data_set

	def read_tfrecord(self, type_):
		if type_ == 't':
			dataset = self.get_data_set(
				filename=os.path.join(self.data_dir, self.train_data_filename + '.tfrecords'),
				shuffle_size=self.batch_size * 10,
				batch_size=self.batch_size,
				prefetch_buffer=self.batch_size * 2)

		if type_ == 'v':
			dataset = self.get_data_set(
				filename=os.path.join(self.data_dir, self.valid_data_filename + '.tfrecords'),
				shuffle_size=self.batch_size * 10,
				batch_size=self.batch_size,
				prefetch_buffer=self.batch_size * 2)

		iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) #dataset.make_one_shot_iterator()
		data = iterator.get_next()

		return data

	#supervised
	def get_data_set_subset(self, filename, shuffle_size, batch_size, prefetch_buffer, subset_size):
		data_set = tf.data.TFRecordDataset(filename)
		data_set = data_set.map(decode)
		data_set = data_set.prefetch(prefetch_buffer)
		data_set = data_set.shuffle(shuffle_size, reshuffle_each_iteration=False)
		data_set = data_set.take(subset_size)
		data_set = data_set.repeat()
		data_set = data_set.batch(batch_size)
		return data_set

	def read_tfrecord_subset(self, type_, subset_size):
		if type_ == 't':
			dataset = self.get_data_set_subset(
				filename=os.path.join(self.data_dir, self.train_data_filename + '.tfrecords'),
				shuffle_size=self.batch_size * 10,
				batch_size=self.batch_size,
				prefetch_buffer=self.batch_size * 2,
				subset_size=subset_size)

		if type_ == 'v':
			dataset = self.get_data_set_subset(
				filename=os.path.join(self.data_dir, self.valid_data_filename + '.tfrecords'),
				shuffle_size=self.batch_size * 10,
				batch_size=self.batch_size,
				prefetch_buffer=self.batch_size * 2,
			    subset_size=subset_size)

		iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) #dataset.make_one_shot_iterator()
		data = iterator.get_next()

		return data

if __name__ == '__main__':
	params = {'patch_size': 28,
			  'sub_patch_size': 20,
			  'batch_size': 100,
			  'supervised': False,
			  'loss_type': "combined",
			  'data_dir': '/Users/hzhang/Documents/project/siamese/sota_comparison/supervised_SimCLR/ova_t/data/random_valid_without_test_with_labels',
			  'train_data_filename': 'train',
			  'valid_data_filename': 'valid'
			  }
	input_gen = generator(**params)
	training_generator = input_gen.generate('t')
	X = next(training_generator)