import os
import pickle
from model_comparisons.util.callbacks import ModelCheckpoint, CSVLogger#, EarlyStopping

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs
import tensorflow as tf
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# from dataset import prepare_dataset
# from augmentations import RandomResizedCrop, RandomColorJitter
from algorithms import SimCLR, MoCo, DINO
from generator import generator
from models import get_feature_encoder, get_feature_encoder_sequential

tf.get_logger().setLevel("WARN")  # suppress info-level logs


model_dir = './model'
# model_name = 'SimCLR_ova_t'



# model_name_list = ['SimCLR_ova_t','SimCLR_ova_pdl1', 'SimCLR_lusc_t', 'SimCLR_hypoxia_fop_fon', "SimCLR_hypoxia_cap_can",
#                    'myeloma', 'gal8_no_intensity']

# model_name_list = ['MoCo_ova_t','MoCo_ova_pdl1', 'MoCo_lusc_t', 'MoCo_hypoxia_fop_fon', "MoCo_hypoxia_cap_can",
#                    'MoCo_myeloma', 'MoCo_gal8_no_intensity']

model_name_list = ["SimCLR_IMC_immune_CD4_CD8"]


initial_epoch = 0
num_of_epoch = 100
patch_size = 28
sub_patch_size = 20
# data_dir = '/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/ova_t/data'
# data_dir_list = ["/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/ova_t/data",
#                 "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/ova_pdl1/data",
#                  "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/lusc_t/data",
#                  "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/hypoxia_fop_fon/data",
#                  "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/hypoxia_cap_can/data",
#                  "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/myeloma/data",
#                  "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/gal8_no_intensity/data"]

data_dir_list = ["/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/IMC_immune_CD4_CD8/data"]

Algorithm = SimCLR
train_data_filename = 'train'
valid_data_filename = 'valid'
batch_size = 256
mini_batch = 5
save_best_only = True
width = 32


for data_i in range(len(model_name_list)):
# for data_i in [5,6]:

    model_name = model_name_list[data_i]
    data_dir = data_dir_list[data_i]

    input_gen = generator(patch_size = patch_size,
                     sub_patch_size = sub_patch_size,
                     batch_size = batch_size,
                     data_path = data_dir)

    # hyperparameters
    # num_epochs = 30
    # steps_per_epoch = 200



    # hyperparameters corresponding to each algorithm
    hyperparams = {
        SimCLR: {"temperature": 0.1},
        MoCo: {"momentum_coeff": 0.99, "temperature": 0.1, "queue_size": 1000},
        DINO: {"momentum_coeff": 0.9, "temperature": 0.1, "sharpening": 0.5},
    }

    # load STL10 dataset
    # batch_size, train_dataset, labeled_train_dataset, test_dataset = prepare_dataset(
    #     steps_per_epoch
    # )

    # select an algorithm

    # architecture
    model = Algorithm(
        contrastive_augmenter=keras.Sequential(
            [
                layers.Input(shape=(patch_size, patch_size, 3)),
                preprocessing.RandomCrop(sub_patch_size, sub_patch_size),
                preprocessing.Rescaling(1 / 255),
                preprocessing.RandomFlip("horizontal")
                # RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3))
                # RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            ],
            name="contrastive_augmenter",
        ),
        classification_augmenter=keras.Sequential(
            [
                layers.Input(shape=(sub_patch_size, sub_patch_size, 3)),
                preprocessing.RandomCrop(sub_patch_size, sub_patch_size),
                preprocessing.Rescaling(1 / 255),
                preprocessing.RandomFlip("horizontal")
                # RandomResizedCrop(scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3))
                # RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ],
            name="classification_augmenter",
        ),
        # encoder=keras.Sequential(
        #     [
        #         layers.Input(shape=(patch_size, patch_size, 3)),
        #         layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
        #         layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
        #         layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
        #         # layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
        #         layers.Flatten(),
        #         layers.Dense(width, activation="relu"),
        #     ],
        #     name="encoder",
        # ),

        encoder=get_feature_encoder_sequential((sub_patch_size, sub_patch_size, 3)),

        projection_head=keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(width, activation="relu"),
                layers.Dense(width),
            ],
            name="projection_head",
        ),
        linear_probe=keras.Sequential(
            [
                layers.Input(shape=(width,)),
                layers.Dense(10),
            ],
            name="linear_probe",
        ),
        **hyperparams[Algorithm],
    )

    # optimizers
    model.compile(
        contrastive_optimizer=keras.optimizers.Adam(),
        probe_optimizer=keras.optimizers.Adam(),
    )


    # run training

    # history = model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)

    def get_train_valid_info():

        if os.path.exists(os.path.join(data_dir, train_data_filename + '_info.npy')):
            param = np.load(os.path.join(data_dir, train_data_filename + '_info.npy'), allow_pickle=True)
            train_num_examples = param.item()['num_examples']
            steps_per_epoch = int(np.ceil(train_num_examples / batch_size)) * int(mini_batch)

            param = np.load(os.path.join(data_dir, valid_data_filename + '_info.npy'), allow_pickle=True)
            valid_num_examples = param.item()['num_examples']
            validation_steps = int(np.ceil(valid_num_examples / batch_size)) * int(mini_batch)

        else:
            import scipy.io as sio
            param = sio.loadmat(os.path.join(data_dir, train_data_filename + '.mat'))
            train_num_examples = param['num_examples'][0][0]
            steps_per_epoch = int(np.ceil(train_num_examples / batch_size)) * int(mini_batch)

            param = sio.loadmat(os.path.join(data_dir, valid_data_filename + '.mat'))
            valid_num_examples = param['num_examples'][0][0]
            validation_steps = int(np.ceil(valid_num_examples / batch_size)) * int(mini_batch)


        print("train img=", train_num_examples)
        print("valid img=", valid_num_examples)

        return steps_per_epoch, validation_steps

    steps_per_epoch, validation_steps = get_train_valid_info()

    if os.path.exists(os.path.join(model_dir, model_name)) is False:
        os.makedirs(os.path.join(model_dir, model_name))

    decimal_places = int(f'{num_of_epoch:e}'.split('e')[-1]) + 1
    model_checkpoint = ModelCheckpoint(os.path.join(model_dir, model_name, model_name +
                                                    '_{epoch:0%id}' % decimal_places), monitor='val_c_loss',
                                                    save_best_only=save_best_only, save_weights_only=False)
    # Callback that streams epoch results to a csv file.

    log_filename = os.path.join(os.path.join(model_dir, model_name, 'train_hist.csv'))
    csv_log = CSVLogger(log_filename, separator=',', append=True)

    # if self.early_stopping:
    #     early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=self.early_stopping, verbose=0, mode='min')
    #     callbacks = [model_checkpoint, csv_log, early_stopping]
    # else:
    callbacks = [model_checkpoint, csv_log]

    H = model.fit(input_gen.generate('t'),
                            shuffle=False,
                            steps_per_epoch=steps_per_epoch,
                            validation_data=input_gen.generate('v'),
                            validation_steps=validation_steps,
                            epochs=num_of_epoch,
                            workers=0,
                            verbose=1,
                            initial_epoch=initial_epoch,
                            callbacks=callbacks)  # , early_stopping

    # tf.keras.models.save_model(
    #     model, os.path.join(model_dir, model_name, model_name +'_model'),
    #     overwrite=True, include_optimizer=True, save_format="tf",
    #     signatures=None, options=None, save_traces=True
    # )

    # save history
    with open("{}.pkl".format(Algorithm.__name__), "wb") as write_file:
        pickle.dump(H.history, write_file)
