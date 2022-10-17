import os
from glob import glob
import pandas as pd

from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')


import random
random.seed(655)

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Activation, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.layers import concatenate


def get_similarity_model():
    img_in = Input(shape = (sub_patch_size, sub_patch_size, 3), name = 'FeatureNet_ImageInput')
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

    img_a_in = Input(shape = (sub_patch_size, sub_patch_size, 3), name = 'ImageA_Input')
    img_b_in = Input(shape = (sub_patch_size, sub_patch_size, 3), name = 'ImageB_Input')
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

def get_features(cell_patches):

    cell_patches_merge = np.reshape(cell_patches, (-1, sub_patch_size, sub_patch_size,3))

    if loss_type == "bce_only":
        feature_layer = Model(inputs=model.inputs, outputs=model.get_layer("merge_features").output)
    else:
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
            if str(im_labels[i]) in cell_classes:
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


def evaluate(labels, pre_labels):
    correct_no = 0
    incorrect_idx = []
    for i in range(len(pre_labels)):
        if pre_labels[i] == labels[i]:
            correct_no += 1
        else:
            incorrect_idx.append(i)

    acc = correct_no / len(pre_labels)
    print('Accuracy:', acc)

    # np.unique(labels[incorrect_idx], return_counts=True)
    # np.unique(np.array(pre_labels)[incorrect_idx], return_counts=True)
    # np.unique(labels, return_counts=True)

    # confusion matrix
    print(classification_report(labels, pre_labels, digits=4))

    cm = confusion_matrix(labels, pre_labels)
    cell_classes_slide = sorted(np.unique(np.concatenate((labels, pre_labels), axis=0)))
    df_cm = pd.DataFrame(cm, index=[i for i in cell_classes_slide],
                         columns=[i for i in cell_classes_slide])
    print(df_cm)
    sns.set(rc={'figure.figsize':(11.0,8.27)}, font_scale=1.4)
    #cmap=sn.cubehelix_palette(light=0.9, as_cmap=True)
    hm = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')  # font size

    return df_cm, hm, incorrect_idx


def get_svm_train_subset(train_features, train_label, test_size, random_state):
    np.random.seed(random_state)
    train_subset = []
    train_label_subset = []
    for c in cell_classes:
        features_c = train_features[np.where(train_label == c)[0]]
        sample_size = int(np.ceil(features_c.shape[0] * test_size))
        index_c = np.random.choice(features_c.shape[0], sample_size, replace=False)
        train_subset_c = features_c[index_c]
        train_label_c = np.array([c] * sample_size)
        train_subset.append(train_subset_c)
        train_label_subset.append(train_label_c)

    train_subset = np.concatenate(train_subset, axis=0)
    train_label_subset = np.concatenate(train_label_subset, axis=0)

    return train_subset, train_label_subset


def SVM_evaluate_simplify(train_npy_path, model, features, labels, labeled_data_ratio, random_state, save_dir):
    print("SVM evaluate")
    slides = glob(os.path.join(train_npy_path, "*" + slide_ext))

    train_cell_patch_list = []
    train_label_list = []
    for slide in slides:
        npy_fs = glob(os.path.join(slide, "*.npy"))
        for npy_f in npy_fs:
            da_npy = np.load(npy_f, allow_pickle=True)
            if len(da_npy[0].shape) == 1 and da_npy[1].shape[0]>0:
                # img_arry = np.concatenate(da_npy[0], axis=0)
                # print(img_arry[0].shape)
                # train_cell_patch_list.append(img_arry)
                img_npy = list(da_npy[0])
                label_npy = list(da_npy[1])
                img_npy = [img_npy[i] for i in range(len(img_npy)) if label_npy[i] in cell_classes]
                label_npy = [label_npy[i] for i in range(len(label_npy)) if label_npy[i] in cell_classes]

                train_cell_patch_list.append(img_npy)
                train_label_list.append(label_npy)

    train_cell_patch = np.concatenate(np.array(train_cell_patch_list, dtype=object), axis=0)
    train_label = np.concatenate(np.array(train_label_list, dtype=object), axis=0)
    train_cell_patch = np.array(train_cell_patch).astype('float32') / 255.
    print(train_cell_patch.shape)

    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1223)
    clf = svm.SVC(kernel='linear')
    class_labels_ = list(np.unique(labels))

    for r in labeled_data_ratio:
        print("\nTraining with %i%% of labeled data:" % (r * 100))

        save_dir_r = os.path.join(save_dir, "ratio_" + str(r))
        if os.path.exists(save_dir_r) is False:
            os.makedirs(save_dir_r)

        output_file = os.path.join(save_dir_r, "all_avg_"+str(r) + '.csv')
        with open(output_file, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(["random_seed","precision","recall","f1-score","support"])

        if r >= 1:
            train_features_subset = get_features(train_cell_patch)

        for r_seed in random_state:
            # print('random seed', r_seed)
            # _, train_cell_patch_subset, _, train_label_subset = train_test_split(train_cell_patch, train_label, test_size=test_size, random_state=2021)
            if r<1:
                train_cell_patch_subset, train_label_subset = get_svm_train_subset(train_cell_patch, train_label, r,
                                                                                 random_state=r_seed)

                train_features_subset = get_features(train_cell_patch_subset)
            else:
                train_label_subset = train_label

            print(np.unique(train_label_subset, return_counts=True))
            print("No. of training samples:", train_label_subset.shape[0])

            clf.fit(train_features_subset, train_label_subset)
            y_pred = clf.predict(features)
            cm = confusion_matrix(labels, y_pred)
            cm = pd.DataFrame(cm, index=[i for i in class_labels_],
                                 columns=[i for i in class_labels_])

            cr = classification_report(labels, y_pred, digits=4, labels=class_labels_, output_dict=True)
            # print(cm)
            print(classification_report(labels, y_pred, digits=3, labels=class_labels_))


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

    return 0

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

    loss_type = "contrast_only"#bce_only, contrast_only, combined
    home_path = "../../ExpDir"
    model_base = data_name + "_unsuper_b_256_opt_adam_" + loss_type + "_ratio_0.3_0.7"
    model_dir = os.path.join(home_path, data_name, "model", model_base)
    model_name = [i for i in os.listdir(model_dir) if i.endswith('.h5')].sort(reverse=True)[0]
    model_path = os.path.join(model_dir, model_name)

    test_npy_path = os.path.join(home_path, data_name, "data", "test")
    train_npy_path = os.path.join(home_path, data_name, "data", "train")

    save_path = os.path.join("../results/self_supervised/", model_base)
    

    patch_size = 28
    sub_patch_size = 20
    labeled_data_ratio = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0]
    random_state = [98, 1006, 665, 379, 507]

    model = get_similarity_model()
    model.load_weights(model_path)

    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    get_tsne = False

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
    print(np.unique(labels_all, return_counts=True))


    cell_features = get_features(cell_patches_all)
    print(cell_features.shape)

    if get_tsne:
        tsne_fig, tsne_features = tsne_represent(cell_features, labels_all, n_iter=500)
        tsne_fig.savefig(os.path.join(save_path, 'tsne.png'))
        np.save(os.path.join(save_path, 'tsne_features.npy'), tsne_features)
        plt.clf()

    SVM_evaluate_simplify(train_npy_path, model, cell_features, labels_all, labeled_data_ratio, random_state, save_path)


