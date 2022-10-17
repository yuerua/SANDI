import os
from glob import glob
import pandas as pd
import numpy as np
import math
import skimage.io as sio
from util.Patches import Patches
from util.data_utils import encode
from util.progressbar import printProgressBar
import tensorflow as tf
from tensorflow.keras import backend as K
import random
from util.move_file import test_split
# from preprocess.npy_to_im import npy_to_im
from distutils.dir_util import copy_tree

class Extractor(object):
    """
    Extract single cell patches from slides
    """
    def __init__(self, opts):
        self.cws_path = opts.cws_path
        self.csv_path = opts.csv_path
        self.slide_ext = opts.slide_ext
        self.save_path = opts.data_path
        self.train_data_filename = opts.train_data_filename
        self.valid_data_filename = opts.valid_data_filename
        self.test_ratio = opts.test_ratio
        self.test_path = opts.test_npy_path

        self.train_valid_split_ratio = opts.train_valid_split_ratio
        self.cell_classes = opts.cell_classes
        self.patch_size = opts.patch_size
        self.sub_patch_size = opts.sub_patch_size
        self.sub_patch_stride = opts.sub_patch_stride
        self.patch_no = int((math.ceil((self.patch_size - self.sub_patch_size) / self.sub_patch_size) + 1)**2)

        self.patch_obj = Patches(
        img_patch_h=opts.sub_patch_size, img_patch_w=opts.sub_patch_size,
        stride_h=opts.sub_patch_stride, stride_w=opts.sub_patch_stride,
        label_patch_h=opts.sub_patch_size, label_patch_w=opts.sub_patch_size)

        if self.cell_classes:
            print("Cell classes:", self.cell_classes)
            self.cell_map = dict(zip(self.cell_classes, range(len(self.cell_classes))))

        self.slides = glob(os.path.join(self.csv_path, '*' + self.slide_ext))
        assert self.slides, "Slides not found"

        if os.path.exists(self.save_path) is False:
            os.makedirs(self.save_path)

        #Separate testing images
        if not self.test_path and self.test_ratio>0:
            self.test_path = os.path.join(self.save_path, "test")
            print("Splitting test samples with ratio %.2f"%self.test_ratio)
            os.makedirs(self.test_path)

            slide_all = glob(os.path.join(self.csv_path, "*" + self.slide_ext))
            self.slides, test_slides = test_split(slide_all, self.test_ratio)

            self.cell_patch_to_npy(self.cws_path, test_slides, self.test_path)
        else:
            self.slides = glob(os.path.join(self.csv_path, "*" + self.slide_ext))
            default_test_path = os.path.join(self.save_path, "test")
            copy_tree(self.test_path, default_test_path)

        #save test images
        cell_img_save_dir = os.path.join(self.save_path, "test_cell_img")
        if os.path.exists(cell_img_save_dir) is False:
            os.makedirs(cell_img_save_dir)

        # npy_to_im(self.test_path, self.slide_ext,cell_img_save_dir, self.patch_obj, self.patch_size, self.cell_classes)


    def cell_patch_to_npy(self, cws_path, slides, save_path):
        #For test images
        #slides: slides csv
        for slide in slides:
            csv_fs = glob(os.path.join(slide, 'Da*'))

            slide_name = os.path.basename(slide)
            save_dir = os.path.join(save_path, slide_name)
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)

            for csv_f in csv_fs:
                im_name = os.path.splitext(os.path.basename(csv_f))[0]
                print('Processing %s from %s'%(im_name, slide_name))
                csv = pd.read_csv(csv_f)

                im_f = os.path.join(cws_path, slide_name, im_name + '.jpg')
                im = sio.imread(im_f)

                pad = np.int(np.ceil(self.patch_size / 2))
                im_pad = np.lib.pad(im, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')

                cell_list = []
                label_list=[]
                for i, row in csv.iterrows():
                    label = str(row['V1'])
                    if label in self.cell_classes or not self.cell_classes:
                        x = row['V2']
                        y = row['V3']

                        x0 = x
                        x1 = x + pad * 2
                        y0 = y
                        y1 = y + pad * 2
                        try:
                            cell_patch = im_pad[y0:y1, x0:x1, :]
                            split_patch = self.patch_obj.extract_patches(cell_patch)
                            cell_list.append(split_patch)
                            label_list.append(label)
                        except:
                            pass

                full_list=[cell_list, label_list]
                save_f = os.path.join(save_dir, im_name + '.npy')
                np.save(save_f, full_list)

    def cell_patch_to_img(self, cws_path, slides, save_path):
        #For test images
        for slide in slides:
            csv_fs = glob(os.path.join(slide, 'Da*'))

            slide_name = os.path.basename(slide)
            save_dir = os.path.join(save_path, slide_name)
            if os.path.exists(save_dir) is False:
                os.makedirs(save_dir)

            for csv_f in csv_fs:
                im_name = os.path.splitext(os.path.basename(csv_f))[0]
                print('Processing %s from %s'%(im_name, slide_name))
                csv = pd.read_csv(csv_f)

                im_f = os.path.join(cws_path, slide_name, im_name + '.jpg')
                im = sio.imread(im_f)

                cell_idx = 0
                for i, row in csv.iterrows():
                    label = str(row['V1'])
                    if label in self.cell_classes or not self.cell_classes:
                        x = row['V2']
                        y = row['V3']
                        pad = np.int(np.ceil(self.patch_size / 2))
                        im_pad = np.lib.pad(im, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
                        x0 = x
                        x1 = x + pad * 2
                        y0 = y
                        y1 = y + pad * 2
                        try:
                            cell_patch = im_pad[y0:y1, x0:x1, :]
                            if np.max(cell_patch) <= 1.:
                                cell_patch = cell_patch * 255.
                            cell_patch = cell_patch.astype('uint8')
                            save_f = os.path.join(save_dir,
                                                  im_name + '_' + str(cell_idx) + '_' + str(label) + '.png')
                            sio.imsave(save_f, cell_patch)
                            cell_idx += 1
                        except:
                            pass

    def write_to_tf_cell_from_npy(self, npy_path):
        #npy structure: [imgs(total_cell_num, 9, sub_patch_size, sub_patch_size, 3), labels(str)]
        print('Writing tfrecords to', self.save_path)
        tf_writer_train = tf.python_io.TFRecordWriter(os.path.join(self.save_path, self.train_data_filename + '.tfrecords'))
        tf_writer_valid = tf.python_io.TFRecordWriter(os.path.join(self.save_path, self.valid_data_filename + '.tfrecords'))
        num_train_examples = 0
        num_valid_examples = 0
        num_patients = 0
        num_da = 0

        cell_type_dict_train = {}
        cell_type_dict_valid = {}

        slides = glob(os.path.join(npy_path, '*' + self.slide_ext))
        Das_no = len(glob(os.path.join(self.csv_path, '*' + self.slide_ext, "*.csv")))
        patient_list = []

        printProgressBar(0, Das_no, prefix='Progress:', suffix='Complete', length=50)

        for slide in slides:
            npy_fs = glob(os.path.join(slide, '*.npy'))

            slide_name = os.path.basename(slide)
            patient_list.append(slide_name)

            for npy_f in npy_fs:
                im_name = os.path.splitext(os.path.basename(npy_f))[0]

                imgs, labels = np.load(npy_f, allow_pickle=True)

                for i in range(imgs.shape[0]):
                    cell_patch = self.patch_obj.merge_patches(np.zeros((self.patch_size, self.patch_size, 3)), imgs[i])
                    cell_patch = cell_patch.astype('uint8')
                    label = labels[i]
                    if str(label) in self.cell_classes:
                        label_idx = int(self.cell_map[str(label)])
                        label_one_hot = K.eval(tf.one_hot(label_idx, len(self.cell_classes), dtype='uint8'))
                        tf_serialized_example = encode(in_feat=cell_patch, labels=label_one_hot)
                        if random.uniform(0, 1) >= self.train_valid_split_ratio:
                            tf_writer_train.write(tf_serialized_example)
                            num_train_examples += 1
                            if not str(label) in cell_type_dict_train.keys():
                                cell_type_dict_train[str(label)] = 1
                            else:
                                cell_type_dict_train[str(label)] += 1
                        else:
                            tf_writer_valid.write(tf_serialized_example)
                            num_valid_examples += 1
                            if not str(label) in cell_type_dict_valid.keys():
                                cell_type_dict_valid[str(label)] = 1
                            else:
                                cell_type_dict_valid[str(label)] += 1

                num_da += 1
                printProgressBar(num_da, Das_no, prefix='Progress:',
                                 suffix='Completed for %s in %s' % (im_name, slide_name), length=50)

            num_patients += 1

        out_dict_train = {'num_examples': num_train_examples, 'patients': patient_list,
                          "num_cell_types": cell_type_dict_train}
        out_dict_valid = {'num_examples': num_valid_examples, 'patients': patient_list,
                          "num_cell_types": cell_type_dict_valid}

        np.save(os.path.join(self.save_path, self.train_data_filename + '_info.npy'), out_dict_train)
        np.save(os.path.join(self.save_path,  self.valid_data_filename + '_info.npy'), out_dict_valid)
        print('\ntrain examples:', num_train_examples, '\nvalid examples:', num_valid_examples)
        tf_writer_train.close()
        tf_writer_valid.close()

    def write_to_tf_cell(self):
        print('Writing tfrecords to', self.save_path)
        tf_writer_train = tf.python_io.TFRecordWriter(os.path.join(self.save_path, self.train_data_filename + '.tfrecords'))
        tf_writer_valid = tf.python_io.TFRecordWriter(os.path.join(self.save_path, self.valid_data_filename + '.tfrecords'))
        num_train_examples = 0
        num_valid_examples = 0
        num_patients = 0
        num_da = 0

        cell_type_dict_train = {}
        cell_type_dict_valid = {}

        #slides = glob(os.path.join(self.csv_path, '*' + self.slide_ext))
        Das_no = len(glob(os.path.join(self.csv_path, '*' + self.slide_ext, "*.csv")))
        patient_list = []
        printProgressBar(0, Das_no, prefix='Progress:', suffix='Complete', length=50)

        for slide in self.slides:
            csv_fs = glob(os.path.join(slide, 'Da*'))

            slide_name = os.path.basename(slide)
            patient_list.append(slide_name)

            for csv_f in csv_fs:
                im_name = os.path.splitext(os.path.basename(csv_f))[0]
                csv = pd.read_csv(csv_f)

                im_f = os.path.join(self.cws_path, slide_name, im_name + '.jpg')
                im = sio.imread(im_f)

                for i, row in csv.iterrows():
                    label = row['V1']
                    x = row['V2']
                    y = row['V3']
                    pad = np.int(np.ceil(self.patch_size / 2))
                    im_pad = np.lib.pad(im, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
                    x0 = x
                    x1 = x + pad * 2
                    y0 = y
                    y1 = y + pad * 2

                    if str(label) in self.cell_classes or not self.cell_classes:
                        label_idx = int(self.cell_map[str(label)])
                        label_one_hot = K.eval(tf.one_hot(label_idx, len(self.cell_classes), dtype='uint8'))

                        cell_patch = im_pad[y0:y1, x0:x1, :]
                        # data = patch_obj.extract_patches(cell_patch)

                        tf_serialized_example = encode(in_feat=cell_patch, labels=np.array(label_one_hot))
                        if random.uniform(0, 1) >= self.train_valid_split_ratio:
                            tf_writer_train.write(tf_serialized_example)
                            num_train_examples += 1

                            if not str(label) in cell_type_dict_train.keys():
                                cell_type_dict_train[str(label)] = 1
                            else:
                                cell_type_dict_train[str(label)] += 1

                        else:
                            tf_writer_valid.write(tf_serialized_example)
                            num_valid_examples += 1

                            if not str(label) in cell_type_dict_valid.keys():
                                cell_type_dict_valid[str(label)] = 1
                            else:
                                cell_type_dict_valid[str(label)] += 1

                num_da += 1
                printProgressBar(num_da, Das_no, prefix='Progress:',
                                 suffix='Completed for %s in %s' % (im_name, slide_name), length=50)

            num_patients += 1

        out_dict_train = {'num_examples': num_train_examples, 'patients': patient_list,
                          "num_cell_types": cell_type_dict_train}
        out_dict_valid = {'num_examples': num_valid_examples, 'patients': patient_list,
                          "num_cell_types": cell_type_dict_valid}
        np.save(os.path.join(self.save_path, self.train_data_filename + '_info.npy'), out_dict_train, allow_pickle=True)
        np.save(os.path.join(self.save_path,  self.valid_data_filename + '_info.npy'), out_dict_valid, allow_pickle=True)
        print('\ntrain examples:', num_train_examples, '\nvalid examples:', num_valid_examples)
        tf_writer_train.close()
        tf_writer_valid.close()
