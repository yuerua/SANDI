import os
import numpy as np

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.python.keras.optimizers import Adam, SGD

from train.model import get_similarity_model, get_supervised_classifier
from train.loss import Loss
from train.generator import generator
from tensorflow.python.keras.losses import binary_crossentropy, categorical_crossentropy
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from util.Patches import Patches
from sklearn.preprocessing import OneHotEncoder

if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Train:

    def __init__(self, opts):
        self.data_dir = opts.data_path
        self.batch_size = opts.batch_size
        self.train_data_filename = opts.train_data_filename
        self.valid_data_filename = opts.valid_data_filename
        self.optimizer_type = opts.optimizer_type
        self.lr = opts.lr
        self.model_name = opts.model_name
        self.save_best_only = opts.save_best_only
        self.pretrain_model = opts.pretrain_model
        self.early_stopping = opts.early_stopping
        self.num_of_epoch = opts.num_of_epoch
        self.mini_batch = opts.mini_batch
        self.model_dir = opts.model_dir
        self.history_dir = opts.history_dir

        self.patch_size = opts.patch_size
        self.sub_patch_size = opts.sub_patch_size
        self.loss_type = opts.loss_type

        self.input_gen = generator(opts)
        self.loss_ = Loss(opts).loss_function()

        self.steps_per_epoch, self.validation_steps, self.train_num_examples, self.valid_num_examples = self.get_train_valid_info()
        self.cell_classes = opts.cell_classes
        self.slide_ext = opts.slide_ext

    def get_train_valid_info(self):
        # param = np.load(os.path.join(self.data_dir, self.train_data_filename + '_info.npy'), allow_pickle=True)
        # train_num_examples = param.item()['num_examples']
        # steps_per_epoch = int(np.ceil(train_num_examples / self.batch_size)) * int(self.mini_batch)
        #
        # param = np.load(os.path.join(self.data_dir, self.valid_data_filename + '_info.npy'), allow_pickle=True)
        # valid_num_examples = param.item()['num_examples']
        # validation_steps = int(np.ceil(valid_num_examples / self.batch_size)) * int(self.mini_batch)

        if os.path.exists(os.path.join(self.data_dir, self.train_data_filename + '_info.npy')):
            param = np.load(os.path.join(self.data_dir, self.train_data_filename + '_info.npy'), allow_pickle=True)
            train_num_examples = param.item()['num_examples']
            steps_per_epoch = int(np.ceil(train_num_examples / self.batch_size)) * int(self.mini_batch)

            param = np.load(os.path.join(self.data_dir, self.valid_data_filename + '_info.npy'), allow_pickle=True)
            valid_num_examples = param.item()['num_examples']
            validation_steps = int(np.ceil(valid_num_examples / self.batch_size)) * int(self.mini_batch)

        else:
            import scipy.io as sio
            param = sio.loadmat(os.path.join(self.data_dir, self.train_data_filename + '.mat'))
            train_num_examples = param['num_examples'][0][0]
            steps_per_epoch = int(np.ceil(train_num_examples / self.batch_size)) * int(self.mini_batch)

            param = sio.loadmat(os.path.join(self.data_dir, self.valid_data_filename + '.mat'))
            valid_num_examples = param['num_examples'][0][0]
            validation_steps = int(np.ceil(valid_num_examples / self.batch_size)) * int(self.mini_batch)

        print("train img=", train_num_examples)
        print("valid img=", valid_num_examples)

        return steps_per_epoch, validation_steps, train_num_examples, valid_num_examples

    def train_and_predict(self):
        model = get_similarity_model((self.sub_patch_size, self.sub_patch_size,3), self.loss_type)
        model.summary()

        # Optimizer
        if self.optimizer_type == 'adam':
            print("Using Adam as optimizer")
            opt_ = Adam(lr=self.lr)
        else:
            #decay = lr / num_of_epoch
            print("Using SGD as optimizer")
            decay = 10**(-6)
            opt_ = SGD(lr=self.lr, momentum=0.9, decay=decay, nesterov=False)

        model.compile(optimizer=opt_, loss=self.loss_, metrics=["accuracy"])

        if os.path.exists(os.path.join(self.model_dir, self.model_name)) is False:
            os.makedirs(os.path.join(self.model_dir, self.model_name))

        if self.pretrain_model:
            model.load_weights(self.pretrain_model)
            pretrain_model_name = os.path.splitext(os.path.basename(self.pretrain_model))[0]
            initial_epoch=int(pretrain_model_name.split("_")[-1])
        else:
            initial_epoch=0

        decimal_places = int(f'{self.num_of_epoch:e}'.split('e')[-1]) + 1
        model_checkpoint = ModelCheckpoint(os.path.join(self.model_dir, self.model_name, self.model_name +
                                                        '_{epoch:0%id}.h5' % decimal_places), monitor='val_loss',
                                                         save_best_only=self.save_best_only)

        # Callback that streams epoch results to a csv file.
        log_filename = os.path.join(os.path.join(self.model_dir, self.model_name, 'train_hist.csv'))
        csv_log = CSVLogger(log_filename, separator=',', append=True)

        if self.early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping, verbose=0, mode='min')
            callbacks = [model_checkpoint, csv_log, early_stopping]
        else:
            callbacks = [model_checkpoint, csv_log]

        H = model.fit_generator(generator=self.input_gen.generate('t'),
                                shuffle=False,
                                steps_per_epoch=self.steps_per_epoch,
                                validation_data=self.input_gen.generate('v'),
                                validation_steps=self.validation_steps,
                                epochs=self.num_of_epoch,
                                workers=0,
                                verbose=1,
                                initial_epoch=initial_epoch,
                                callbacks=callbacks) #, early_stopping


        return H

    def train_and_predict_subset_supervised(self, size_ratio):
        model = get_supervised_classifier((self.patch_size, self.patch_size,3), len(self.cell_classes))
        subset_size_train = int(self.train_num_examples * size_ratio)
        subset_size_valid = int(self.valid_num_examples * size_ratio)

        # Optimizer
        if self.optimizer_type == 'adam':
            print("Using Adam as optimizer")
            opt_ = Adam(lr=self.lr)
        else:
            #decay = lr / num_of_epoch
            print("Using SGD as optimizer")
            decay = 10**(-6)
            opt_ = SGD(lr=self.lr, momentum=0.9, decay=decay, nesterov=False)

        if len(self.cell_classes) == 2:
            loss_ = binary_crossentropy
        else:
            loss_ = categorical_crossentropy

        model.compile(optimizer=opt_, loss=loss_, metrics=["accuracy"])

        if os.path.exists(os.path.join(self.model_dir, "super_subset_" + str(size_ratio) + "_" + self.model_name)) is False:
            os.makedirs(os.path.join(self.model_dir, "super_subset_" + str(size_ratio) + "_" + self.model_name))

        if self.pretrain_model:
            model.load_weights(self.pretrain_model)
            pretrain_model_name = os.path.splitext(os.path.basename(self.pretrain_model))[0]
            initial_epoch=int(pretrain_model_name.split("_")[-1])
        else:
            initial_epoch=0

        decimal_places = int(f'{self.num_of_epoch:e}'.split('e')[-1]) + 1
        model_checkpoint = ModelCheckpoint(os.path.join(self.model_dir, "super_subset_" + str(size_ratio) + "_" + self.model_name,
                                                        "super_subset_" + str(size_ratio) + "_" + self.model_name +
                                                        '.h5'), monitor='val_loss',
                                                         save_best_only=self.save_best_only)

        # Callback that streams epoch results to a csv file.
        log_filename = os.path.join(os.path.join(self.model_dir, "super_subset_" + str(size_ratio) + "_" + self.model_name,
                                                 'train_hist.csv'))
        csv_log = CSVLogger(log_filename, separator=',', append=True)

        if self.early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping, verbose=0, mode='min')
            callbacks = [model_checkpoint, csv_log, early_stopping]
        else:
            callbacks = [model_checkpoint, csv_log]

        H = model.fit_generator(generator=self.input_gen.generate_subset('t', subset_size_train),
                                shuffle=True,
                                steps_per_epoch=int(np.ceil(self.steps_per_epoch / self.mini_batch * size_ratio)),
                                validation_data=self.input_gen.generate_subset('v', subset_size_valid),
                                validation_steps=int(np.ceil(self.validation_steps / self.mini_batch * size_ratio)),
                                epochs=self.num_of_epoch,
                                workers=0,
                                verbose=1,
                                initial_epoch=initial_epoch,
                                callbacks=callbacks) #, early_stopping


        return H

    def train_and_predict_subset_supervised_balance(self, size_ratio, random_seed):
        npy_data_path = os.path.join(self.data_dir, "train")

        slides = glob(os.path.join(npy_data_path, "*" + self.slide_ext))

        train_cell_patch_list = []
        train_label_list = []
        for slide in slides:
            npy_fs = glob(os.path.join(slide, "*.npy"))
            for npy_f in npy_fs:
                da_npy = np.load(npy_f, allow_pickle=True)
                if len(da_npy[0].shape) == 1 and da_npy[1].shape[0] > 0:
                    # img_arry = np.concatenate(da_npy[0], axis=0)
                    # print(img_arry[0].shape)
                    # train_cell_patch_list.append(img_arry)
                    img_npy = list(da_npy[0])
                    label_npy = list(da_npy[1])

                    img_npy = [img_npy[i] for i in range(len(img_npy)) if label_npy[i] in self.cell_classes]
                    label_npy = [label_npy[i] for i in range(len(label_npy)) if label_npy[i] in self.cell_classes]

                    train_cell_patch_list.append(img_npy)
                    train_label_list.append(label_npy)

        all_cell_patch = np.concatenate(np.array(train_cell_patch_list, dtype=object), axis=0)
        all_label = np.concatenate(np.array(train_label_list, dtype=object), axis=0)
        print(np.unique(all_label, return_counts=True))

        patch_obj = Patches(
            img_patch_h=self.sub_patch_size, img_patch_w=self.sub_patch_size,
            stride_h=4, stride_w=4,
            label_patch_h=self.sub_patch_size, label_patch_w=self.sub_patch_size)

        all_cell_patch = [patch_obj.merge_patches(np.zeros((self.patch_size, self.patch_size, 3)), i) for i in
                                  all_cell_patch]
        all_cell_patch = np.array(all_cell_patch)

        all_label_one_hot = all_label.reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(all_label_one_hot)

        # if size_ratio>=1:
        #     subset_cell_patch_subset = all_cell_patch
        #     subset_label_subset = all_label
        #
        # else:
        #     #default random seed 2021
        #     _, subset_cell_patch_subset, _, subset_label_subset = train_test_split(all_cell_patch, all_label,
        #                                                         test_size=size_ratio, random_state=random_seed, stratify=all_label)


        # X_train, X_test, y_train, y_test = train_test_split(subset_cell_patch_subset, subset_label_subset,
        #                                                     test_size=0.3, random_state=random_seed, stratify=subset_label_subset)

        X_train, X_test, y_train, y_test = train_test_split(all_cell_patch, all_label,
                                                            test_size=size_ratio, random_state=random_seed, stratify=all_label)
        print(np.unique(y_train, return_counts=True))
        print(np.unique(y_test, return_counts=True))

        training_datagen = ImageDataGenerator(
            rescale=1. / 255,
            horizontal_flip=True)

        validation_datagen = ImageDataGenerator(rescale=1. / 255)

        training_datagen.fit(X_train)
        validation_datagen.fit(X_test)
        training_labels = enc.transform(np.array(y_train).reshape(-1,1)).toarray()
        testing_labels = enc.transform(np.array(y_test).reshape(-1,1)).toarray()


        model = get_supervised_classifier((self.patch_size, self.patch_size,3), len(self.cell_classes))
        # subset_size_train = int(self.train_num_examples * size_ratio)
        # subset_size_valid = int(self.valid_num_examples * size_ratio)

        # Optimizer
        if self.optimizer_type == 'adam':
            print("Using Adam as optimizer")
            opt_ = Adam(lr=self.lr)
        else:
            #decay = lr / num_of_epoch
            print("Using SGD as optimizer")
            decay = 10**(-6)
            opt_ = SGD(lr=self.lr, momentum=0.9, decay=decay, nesterov=False)

        if len(self.cell_classes) == 2:
            loss_ = binary_crossentropy
        else:
            loss_ = categorical_crossentropy

        model.compile(optimizer=opt_, loss=loss_, metrics=["accuracy"])

        if os.path.exists(os.path.join(self.model_dir, "270922_balance_super_subset")) is False:
            os.makedirs(os.path.join(self.model_dir, "270922_balance_super_subset"))

        if self.pretrain_model:
            model.load_weights(self.pretrain_model)
            pretrain_model_name = os.path.splitext(os.path.basename(self.pretrain_model))[0]
            initial_epoch=int(pretrain_model_name.split("_")[-1])
        else:
            initial_epoch=0

        decimal_places = int(f'{self.num_of_epoch:e}'.split('e')[-1]) + 1
        model_checkpoint = ModelCheckpoint(os.path.join(self.model_dir, "270922_balance_super_subset",
                                                        "supervised_" + str(size_ratio) + "_" + str(random_seed) + '.h5'), monitor='val_loss',
                                                         save_best_only=self.save_best_only)

        # Callback that streams epoch results to a csv file.
        log_filename = os.path.join(os.path.join(self.model_dir, "270922_balance_super_subset",
                                                 "supervised_" + str(size_ratio) + "_" + str(random_seed) + '_train_hist.csv'))
        csv_log = CSVLogger(log_filename, separator=',', append=True)

        if self.early_stopping:
            early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping, verbose=0, mode='min')
            callbacks = [model_checkpoint, csv_log, early_stopping]
        else:
            callbacks = [model_checkpoint, csv_log]

        H = model.fit_generator(generator=training_datagen.flow(
                                    X_train,
                                    y=training_labels,
                                    batch_size=self.batch_size),
                                shuffle=True,
                                steps_per_epoch=int(np.ceil(self.steps_per_epoch / self.mini_batch * size_ratio)),
                                validation_data=validation_datagen.flow(
                                    X_test,
                                    y=testing_labels,
                                    batch_size=self.batch_size),
                                validation_steps=int(np.ceil(self.validation_steps / self.mini_batch * size_ratio)),
                                epochs=self.num_of_epoch,
                                workers=0,
                                verbose=1,
                                initial_epoch=initial_epoch,
                                callbacks=callbacks) #, early_stopping

        return H



#if __name__ == '__main__':
    # data_dir = '/Users/hzhang/Documents/project/SANDI/dataset/hypoxia/data/data_20_4_can_cap_from_npy'
    # train_data_filename = 'train'
    # valid_data_filename = 'valid'
    # input_shape = (20, 20, 3)
    # pretrain_model = ''
    # dataset_name = "hypoxia_can_cap"
    # batch_size = 256
    # lr = 0.001
    # num_of_epoch=100
    # mini_batch = 5
    # optimizer_type = 'adam'
    # supervised = False
    # save_best_only = True
    # loss_type = "combined"# ["bce_only", "contrast_only", "combined"]
    # bce_w1 = 0.7
    # bce_w0 = 0.3
    #
    # model_name = get_model_name(dataset_name)
    #
    #
    # params = {'patch_size': 28,
    #           'sub_patch_size': input_shape[0],
    #           'batch_size': batch_size,
    #           'supervised': supervised,
    #           "loss_type": loss_type,
    #           'data_dir': data_dir,
    #           'train_data_filename': 'train',
    #           'valid_data_filename': 'valid'
    #           }
    #
    #
    #
    # K.clear_session()
    # K.get_session().close()
    # K.set_session(tf.Session())
    # K.get_session().run(tf.global_variables_initializer())
    #
    # input_gen = generator(**params)
    #
    #
    # H = train_and_predict()
    # plot_training_curves(output_dir='./history', H=H)





