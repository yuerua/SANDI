import os
from glob import glob
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from util.Patches import Patches
import csv

import matplotlib.pyplot as plt
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')

import random
random.seed(655)

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout, Lambda


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

def get_features(cell_patches):

    cell_patches_merge = np.reshape(cell_patches, (-1, sub_patch_size, sub_patch_size,3))

    feature_layer = Model(inputs=model.inputs, outputs=model.get_layer("Contrast").output)

    features = feature_layer.predict([cell_patches_merge, cell_patches_merge])[:, :32]
    features = np.reshape(features, (cell_patches.shape[0], -1))
    return features


def get_slide_batch(test_slide):
    cell_patches = []
    labels = []
    im_idex = []
    im_names = []
    im_fs = sorted(glob(os.path.join(slide, 'Da*.npy')))
    for im_f in im_fs:
        im_name = os.path.splitext(os.path.basename(im_f))[0]
        # print(im_name)
        im_cell_patches, im_labels = np.load(im_f, allow_pickle=True)
        im_idex_count = 0
        for i in range(im_cell_patches.shape[0]):
            if str(im_labels[i]) in cell_classes or str(im_labels[i]) == "pstat":
                im_idex_count += 1
                cell_patches.append(im_cell_patches[i])
                labels.append(im_labels[i])

        im_idex.append(im_idex_count)
        im_names.append(im_name)

    im_name_list = [idx * [name] for idx, name in zip(im_idex, im_names)]
    im_name_list = sum(im_name_list, [])

    cell_patches = np.stack(cell_patches, 0)
    if np.max(cell_patches) > 1.:
        cell_patches = cell_patches.astype('float32') / 255.
    labels = np.array(labels)
    # np.unique(labels, return_counts=True)
    return cell_patches, labels, im_name_list



def tsne_represent(features, labels, n_iter=500):
    print('using tsne')
    # tsne
    tsne_obj = TSNE(n_components=2,
                    init='pca',
                    random_state=101,
                    method='barnes_hut',
                    n_iter=n_iter,
                    verbose=0)

    tsne_features = tsne_obj.fit_transform(features)

    #merge_pixel = np.reshape(cell_patches,(cell_patches.shape[0], -1))
    #tsne_features = tsne_obj.fit_transform(merge_pixel)

    #cell_classes = np.unique(labels).tolist()
    #colors = plt.cm.rainbow(np.linspace(0, 1, len(cell_classes)))
    #print(plt.cm.cmap_d.keys())
    cell_classes = np.unique(np.array(labels))
    print(cell_classes)
    colors = plt.get_cmap('twilight')(np.linspace(0, 1, len(cell_classes)))

    f = plt.figure(figsize=(10, 8))
    for idex, (c_color, c_label) in enumerate(zip(colors, cell_classes)):
        plt.scatter(tsne_features[np.where(labels == c_label), 0],
                    tsne_features[np.where(labels == c_label), 1],
                    marker='o',
                    # color=c_color,
                    linewidth=1,
                    alpha=0.8,
                    label=c_label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on Testing Samples')
    plt.legend(loc='best')
    #plt.pause(5)

    return f, tsne_features

if __name__ == "__main__":

    data_name = "ova_t"
    slide_ext = ".ndpi"
    cell_classes = sorted(['cd4', 'cd8', 'foxp3', 'pd1', 'pd1cd8', 'pd1cd4'])

    # data_name = "lusc_t"
    # slide_ext = ".ndpi"
    # cell_classes = sorted(["cd4", "cd8", "foxp3", "uc"])

    # data_name = "dcis_fop_fon"
    # slide_ext = ".svs"
    # cell_classes = sorted(["fop", "fon"])

    # data_name = "myeloma"
    # slide_ext = ".ndpi"
    # cell_classes = sorted(["CD8", "CD4", "FOXP3+CD4+"])

    # data_name = "IMC_immune_CD4_CD8"
    # slide_ext = ".ndpi"
    # cell_classes = sorted(["Th", "Tc"])

    home_path = "../../ExpDir"
    model_path = os.path.join(home_path, data_name, "model")
    test_npy_path = os.path.join(home_path, data_name, "data", "test")

    save_path = os.path.join("../results/supervised/", "balance_super_subset_" + data_name)

    patch_size = 28
    sub_patch_size = 20
    labeled_data_ratio = [0.1, 0.2, 0.3, 1.0]
    random_seed = [98, 1006, 665, 379, 507]


    patch_obj = Patches(
        img_patch_h=sub_patch_size, img_patch_w=sub_patch_size,
        stride_h=4, stride_w=4,
        label_patch_h=sub_patch_size, label_patch_w=sub_patch_size)

    model = get_supervised_classifier((patch_size, patch_size, 3), len(cell_classes))

    get_tsne =  False

    test_slide = sorted(glob(os.path.join(test_npy_path, '*' + slide_ext)))

    cell_patches_all = []
    labels_all = []
    for slide in test_slide:
        slide_name = os.path.basename(slide)
        slide_name_f = os.path.splitext(slide_name)[0]

        cell_patches, labels, im_name_list = get_slide_batch(slide)

        cell_patches_all.append(cell_patches)
        labels_all.append(labels)


    if len(test_slide) == 1:
        cell_patches_all = np.array(cell_patches_all)[0]
    else:
        cell_patches_all = np.concatenate(np.array(cell_patches_all, dtype=object), axis=0)

    labels_all = np.concatenate(np.array(labels_all, dtype=object), axis=0)

    print('Total number of cells', labels_all.shape[0])
    # print(cell_patches_all.shape)
    print(np.unique(labels_all, return_counts=True))

    # f = lambda x: patch_obj.merge_patches(np.zeros((patch_size, patch_size, 3)), x)
    # cell_patches_all_merge = np.vectorize(f)(cell_patches_all)
    cell_patches_all_merge = [patch_obj.merge_patches(np.zeros((patch_size, patch_size, 3)), i) for i in cell_patches_all]
    cell_patches_all_merge = np.array(cell_patches_all_merge)

    for r in labeled_data_ratio:

        save_dir_r = os.path.join(save_path, "ratio_" + str(r))
        if os.path.exists(save_dir_r) is False:
            os.makedirs(save_dir_r)

        output_file = os.path.join(save_dir_r, "all_avg_"+str(r) + '.csv')
        with open(output_file, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(["random_seed","precision","recall","f1-score","support"])

        for r_seed in random_seed:
            print("\nTraining with %i%% of labeled data:" % (r * 100))
            # model_name = "balance_super_subset_" + str(r) + "_" + data_name + "_super_b_256_opt_adam_combined_ratio_0.7_0.3"
            model_name = "supervised_" + str(r) + "_" + str(r_seed) + ".h5"
            model_path_r = os.path.join(model_path, "balance_super_subset", model_name)
            print(model_path_r)
            model.load_weights(model_path_r)

            pre_score = model.predict(cell_patches_all_merge)
            pre_labels = pre_score.argmax(-1)
            pre_labels = [cell_classes[i] for i in pre_labels]

            cm = confusion_matrix(labels_all, pre_labels)
            cm = pd.DataFrame(cm, index=[i for i in cell_classes],
                              columns=[i for i in cell_classes])
            cr = classification_report(labels_all, pre_labels, digits=4, labels=cell_classes, output_dict=True)
            print(classification_report(labels_all, pre_labels, digits=4, labels=cell_classes))

            cm.to_csv(os.path.join(save_dir_r, 'random_seed_' + str(r_seed) + '_cm.csv'))
            cr = pd.DataFrame(cr).transpose()
            cr.to_csv(os.path.join(save_dir_r, 'random_seed_' + str(r_seed) + '_cr.csv'))

            with open(output_file, "a+", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([r_seed] + cr.tail(1).values.tolist()[0])

        df_result_r = pd.read_csv(output_file)
        df_result_r.loc['mean'] = df_result_r.mean()
        print(df_result_r)
        df_result_r.to_csv(output_file)
