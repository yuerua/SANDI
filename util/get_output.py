# Description: Generate predictions

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.keras.models import Model
from glob import glob
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
import math
import skimage.io as sio
from util.Patches import Patches

class Output(object):
    def __init__(self, cell_patches,
                 ref_patches,
                 labels,
                 ref_labels,
                 model,

                 patch_size,
                 sub_patch_size,
                 sub_patch_stride,
                 cell_classes,

                 mode="compare_feature_dist",
                 loss_type="combined",
                 verbose=0,
                 **kwargs):

        self.cell_patches=cell_patches
        self.ref_patches=ref_patches
        self.labels=labels
        self.ref_labels=ref_labels
        self.model = model

        self.mode=mode
        self.loss_type=loss_type
        self.verbose=verbose

        self.patch_size=patch_size
        self.sub_patch_size=sub_patch_size
        self.sub_patch_stride=sub_patch_stride
        self.cell_classes=cell_classes
        self.patch_no = int((math.ceil((self.patch_size - self.sub_patch_size) / self.sub_patch_stride) + 1)**2)


        self.patch_pairs = [[x, y] for x in range(self.patch_no) for y in range(self.patch_no)]
        self.input_shape=(self.sub_patch_size, self.sub_patch_size, 3)
        self.patch_pairs = np.transpose(np.array(self.patch_pairs)).tolist()


        self.pre_labels = []
        self.pre_score = []
        self.pre_batch_score = []
        self.pre_distance = []

        self.patch_obj = Patches(
        img_patch_h=self.sub_patch_size, img_patch_w=self.sub_patch_size,
        stride_h=self.sub_patch_stride, stride_w=self.sub_patch_stride,
        label_patch_h=self.sub_patch_size, label_patch_w=self.sub_patch_size)

    def output(self, mode):

        self.mode = mode

        if self.mode == "compare_feature_dist":
            result = self.compare_feature_dist()

        elif self.mode == "compare_max":
            result = self.compare_max()

        elif self.mode == "compare_middle":
            result = self.compare_middle()

        elif self.mode == "compare_distance":
            result = self.compare_distance()
        else:
            print("Using SVM")
            result = self.compare_SVM()

        return result

    def get_features(self, cell_patches):

        cell_patches_merge = np.reshape(cell_patches, (-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        if self.loss_type == "bce_only":
            feature_layer = Model(inputs=self.model.inputs, outputs=self.model.get_layer("merge_features").output)
        else:
            feature_layer = Model(inputs=self.model.inputs, outputs=self.model.get_layer("Contrast").output)

        features = feature_layer.predict([cell_patches_merge, cell_patches_merge])[:, :32]
        features = np.reshape(features, (cell_patches.shape[0], -1))
        return features

    def get_output_ref(self, cell_patches, labels, ref_patches, ref_labels):
        patch_pairs = [[x, y] for x in range(self.patch_no) for y in range(self.patch_no)]
        patch_pairs = np.transpose(np.array(patch_pairs)).tolist()
        pre_labels = []
        pre_score = []
        pre_batch_score = []
        for i in range(cell_patches.shape[0]):
            im_1 = cell_patches[i]  # 9*20*20
            cell_score = []
            for ref_i in range(ref_patches.shape[0]):
                im_0 = ref_patches[ref_i]
                # im_0 = np.reshape(im_0, (-1, input_shape[0], input_shape[1], input_shape[2]))

                pair_batch_0 = im_0[patch_pairs[0]]
                pair_batch_1 = im_1[patch_pairs[1]]

                batch_score_s = self.model.predict([pair_batch_0, pair_batch_1])
                batch_score_r = self.model.predict([pair_batch_1, pair_batch_0])

                batch_score = (batch_score_s + batch_score_r) / 2.
                max_score = np.max(batch_score)

                cell_score.append(max_score)
                pre_batch_score.append(batch_score)

            cell_score = np.array(cell_score)
            cell_label = ref_labels[np.where(cell_score == np.max(cell_score))[0][0]]

            print(i, 'gt:', labels[i], 'pred:', cell_label, 'max_score:', np.round(np.max(cell_score), 3))
            pre_labels.append(cell_label)
            pre_score.append(cell_score)

        return np.stack(pre_labels, 0), np.stack(pre_score, 0), np.stack(pre_batch_score, 0)

    def get_ref_diagnosis(self, ref_labels, pre_labels_ref, save_path):
        print('Diagnosing reference set')
        if len(np.where(pre_labels_ref != ref_labels)[0]):
            if os.path.exists(os.path.join(save_path, "bad_ref")) is False:
                os.makedirs(os.path.join(save_path, "bad_ref"))
            for bad_ref_i in np.where(pre_labels_ref != ref_labels)[0]:
                save_bad_f = os.path.join(save_path, "bad_ref", str(bad_ref_i) + \
                                          '_%s_%s.png' % (ref_labels[bad_ref_i], pre_labels_ref[bad_ref_i]))
                bad_ref = self.patch_obj.merge_patches(np.zeros((self.patch_size, self.patch_size, 3)), self.ref_patches[bad_ref_i])
                bad_ref = (bad_ref * 255.).astype('uint8')
                sio.imsave(save_bad_f, bad_ref)

        assert (pre_labels_ref == ref_labels).all(), "Bad ref img found for %s" % \
                                                     ref_labels[np.where(pre_labels_ref != ref_labels)]
        print('Reference passed')

    def compare_feature_dist(self):
        ref_features = self.get_features(self.ref_patches)
        features = self.get_features(self.cell_patches)

        cell_dist = []
        for cell_i in range(features.shape[0]):
            cell_f = np.expand_dims(features[cell_i], axis=0)
            pair_distance = pairwise_distances(cell_f, ref_features)[0]
            cell_dist.append(pair_distance)
            cell_label = self.ref_labels[np.where(pair_distance == np.min(pair_distance))[0][0]]

            self.pre_distance.append(pair_distance)

            min_dist = np.min(pair_distance, axis=0)
            self.pre_labels.append(cell_label)
            self.pre_score.append(min_dist)
            self.pre_batch_score.append(pair_distance)

            if self.verbose == 1:
                print(cell_i, 'gt:', self.labels[cell_i], 'pred:', cell_label,
                      'pair distance:', min_dist)

        return np.stack(self.pre_labels, 0), np.stack(self.pre_score, 0), \
               np.stack(self.pre_batch_score, 0), np.stack(self.pre_distance, 0)

    def compare_max(self):

        pre_labels_ref, pre_score_ref, pre_batch_score_ref = \
            self.get_output_ref(self.ref_patches, self.ref_labels, self.ref_patches, self.ref_labels)

        pre_batch_score_ref = np.reshape(pre_batch_score_ref, (-1, self.ref_labels.shape[0], self.patch_no ** 2))

        for i in range(self.cell_patches.shape[0]):
            im_1 = self.cell_patches[i] # 9*20*20
            cell_score = []
            cell_batch_score = []
            for ref_i in range(self.ref_patches.shape[0]):
                im_0 = self.ref_patches[ref_i]
                # im_0 = np.reshape(im_0, (-1, input_shape[0], input_shape[1], input_shape[2]))

                pair_batch_0 = im_0[self.patch_pairs[0]]
                pair_batch_1 = im_1[self.patch_pairs[1]]

                if self.loss_type == "bce_only":
                    batch_score_s = self.model.predict([pair_batch_0, pair_batch_1])
                    batch_score_r = self.model.predict([pair_batch_1, pair_batch_0])
                else:
                    batch_score_s = self.model.predict([pair_batch_0, pair_batch_1])[1]
                    batch_score_r = self.model.predict([pair_batch_1, pair_batch_0])[1]

                batch_score = (batch_score_s + batch_score_r) / 2.
                max_score = np.max(batch_score)

                cell_batch_score.append(batch_score)
                cell_score.append(max_score)
                self.pre_batch_score.append(batch_score)

            cell_batch_score = np.stack(cell_batch_score, 0)
            cell_batch_score = np.reshape(cell_batch_score, (cell_batch_score.shape[0], cell_batch_score.shape[1]))
            cell_score = np.array(cell_score)
            cell_label = self.ref_labels[np.where(cell_score == np.max(cell_score))[0][0]]
            # distance
            cell_vec = np.concatenate(cell_batch_score).ravel()
            cell_vec = np.expand_dims(cell_vec, axis=0)
            ref_vec = np.concatenate(pre_batch_score_ref[np.where(cell_score == np.max(cell_score))[0][0]]).ravel()
            ref_vec = np.expand_dims(ref_vec, axis=0)
            pair_distance = pairwise_distances(cell_vec, ref_vec)[0][0]

            # if detect_neg:
            #     if np.max(cell_score) < neg_th:
            #         cell_label = 'neg'

            self.pre_distance.append(pair_distance)
            self.pre_labels.append(cell_label)
            self.pre_score.append(cell_score)
            if self.verbose == 1:
                print(i, 'gt:', self.labels[i], 'pred:', cell_label, 'max_score:', np.round(np.max(cell_score), 3),
                      'pair distance:', pair_distance)

        return np.stack(self.pre_labels, 0), np.stack(self.pre_score, 0), \
               np.stack(self.pre_batch_score, 0), np.stack(self.pre_distance, 0)

    def compare_middle(self):

        pre_labels_ref, pre_score_ref, pre_batch_score_ref = \
            self.get_output_ref(self.ref_patches, self.ref_labels, self.ref_patches, self.ref_labels)

        for i in range(self.cell_patches.shape[0]):
            im_1 = self.cell_patches[i]  # 9*20*20
            cell_score = []
            cell_batch_score = []

            for ref_i in range(self.ref_patches.shape[0]):
                im_0 = self.ref_patches[ref_i]
                # im_0 = np.reshape(im_0, (-1, input_shape[0], input_shape[1], input_shape[2]))
                middle_idx = int(np.floor(self.patch_no / 2))
                pair_batch_0 = np.expand_dims(im_0[middle_idx], axis=0)
                pair_batch_1 = np.expand_dims(im_1[middle_idx], axis=0)

                if self.loss_type == "bce_only":
                    batch_score_s = self.model.predict([pair_batch_0, pair_batch_1])
                    batch_score_r = self.model.predict([pair_batch_1, pair_batch_0])
                else:
                    batch_score_s = self.model.predict([pair_batch_0, pair_batch_1])[1]
                    batch_score_r = self.model.predict([pair_batch_1, pair_batch_0])[1]

                batch_score = (batch_score_s + batch_score_r) / 2.
                max_score = np.max(batch_score)

                cell_batch_score.append(batch_score)
                cell_score.append(max_score)
                self.pre_batch_score.append(batch_score)

            cell_batch_score = np.stack(cell_batch_score, 0)
            cell_batch_score = np.reshape(cell_batch_score, (cell_batch_score.shape[0], cell_batch_score.shape[1]))
            cell_score = np.array(cell_score)
            cell_label = self.ref_labels[np.where(cell_score == np.max(cell_score))[0][0]]
            # distance
            cell_vec = np.concatenate(cell_batch_score).ravel()
            cell_vec = np.expand_dims(cell_vec, axis=0)
            ref_vec = pre_score_ref[np.where(cell_score == np.max(cell_score))[0][0]]
            ref_vec = np.expand_dims(ref_vec, axis=0)
            pair_distance = pairwise_distances(cell_vec, ref_vec)[0][0]

            # if detect_neg:
            #     if np.max(cell_score) < neg_th:
            #         cell_label = 'neg'

            self.pre_distance.append(pair_distance)

            self.pre_labels.append(cell_label)
            self.pre_score.append(cell_score)

            if self.verbose == 1:
                print(i, 'gt:', self.labels[i], 'pred:', cell_label, 'max_score:', np.round(np.max(cell_score), 3),
                      'pair distance:', pair_distance)

        return np.stack(self.pre_labels, 0), np.stack(self.pre_score, 0), \
               np.stack(self.pre_batch_score, 0), np.stack(self.pre_distance, 0)

    def compare_distance(self):

        pre_labels_ref, pre_score_ref, pre_batch_score_ref = \
            self.get_output_ref(self.ref_patches, self.ref_labels, self.ref_patches, self.ref_labels)

        pre_batch_score_ref = np.reshape(pre_batch_score_ref, (-1, self.ref_labels.shape[0], self.patch_no ** 2))

        for i in range(self.cell_patches.shape[0]):
            im_1 = self.cell_patches[i]  # 9*20*20
            cell_score = []
            cell_batch_score = []

            for ref_i in range(self.ref_patches.shape[0]):
                im_0 = self.ref_patches[ref_i]
                #im_0 = np.reshape(im_0, (-1, input_shape[0], input_shape[1], input_shape[2]))

                pair_batch_0 = im_0[self.patch_pairs[0]]
                pair_batch_1 = im_1[self.patch_pairs[1]]

                if self.loss_type == "bce_only":
                    batch_score_s = self.model.predict([pair_batch_0, pair_batch_1])
                    batch_score_r = self.model.predict([pair_batch_1, pair_batch_0])
                else:
                    batch_score_s = self.model.predict([pair_batch_0, pair_batch_1])[1]
                    batch_score_r = self.model.predict([pair_batch_1, pair_batch_0])[1]

                batch_score = (batch_score_s + batch_score_r) / 2.
                max_score = np.max(batch_score)

                cell_batch_score.append(batch_score)
                cell_score.append(max_score)
                self.pre_batch_score.append(batch_score)

            #distance
            cell_vec = np.concatenate(cell_batch_score).ravel()
            cell_vec = np.expand_dims(cell_vec, axis=0)

            cell_dist = []
            for ref_i in range(self.ref_patches.shape[0]):
                ref_vec = np.concatenate(pre_batch_score_ref[ref_i]).ravel()
                ref_vec = np.expand_dims(ref_vec, axis=0)
                pair_distance = pairwise_distances(cell_vec, ref_vec)[0][0]
                cell_dist.append(pair_distance)

            cell_label = self.ref_labels[np.where(cell_dist == np.min(cell_dist))[0][0]]

            self.pre_labels.append(cell_label)
            self.pre_score.append(cell_score)
            self.pre_distance.append(np.min(cell_dist))

            if self.verbose == 1:
                print(i, 'gt:', self.labels[i], 'pred:', cell_label, 'max_score:', np.round(np.max(cell_score), 3),
                      'pair distance:', pair_distance)

        return np.stack(self.pre_labels, 0), np.stack(self.pre_score, 0), \
               np.stack(self.pre_batch_score, 0), np.stack(self.pre_distance, 0)

    def compare_SVM(self):
        clf = svm.SVC(kernel='linear')
        ref_features = self.get_features(self.ref_patches)
        features = self.get_features(self.cell_patches)
        clf.fit(ref_features, self.ref_labels)
        y_pred = clf.predict(features)
        # cm = confusion_matrix(self.labels, y_pred)
        # print("SVM with reference set")
        # print(cm)
        # print(classification_report(self.labels, y_pred, digits=4))

        p = np.array(clf.decision_function(features))  # decision is a voting function
        prob = np.exp(p) / np.sum(np.exp(p), axis=-1, keepdims=True)

        self.pre_labels = y_pred
        self.pre_score = np.max(prob, axis=-1)
        self.pre_batch_score = prob
        self.pre_distance = 1./prob

        return self.pre_labels, self.pre_score, \
               self.pre_batch_score, self.pre_distance

    def get_svm_train_subset(self, train_features, train_label, test_size, cell_classes, random_state):
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

    def SVM_evaluate(self, train_npy_path, slide_ext, opt_random_state, cell_classes, opt_test_size):
        print("SVM evaluate")
        slides = glob(os.path.join(train_npy_path, "*" + slide_ext))

        train_cell_patch_list = []
        train_label_list = []
        for slide in slides:
            npy_fs = glob(os.path.join(slide, "*.npy"))
            for npy_f in npy_fs:
                da_npy = np.load(npy_f, allow_pickle=True)
                train_cell_patch_list.append(list(da_npy[0]))
                train_label_list.append(list(da_npy[1]))

        train_cell_patch = np.concatenate(train_cell_patch_list, axis=0)
        train_cell_patch = np.reshape(train_cell_patch, (-1, self.patch_no, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        train_cell_patch = np.array(train_cell_patch).astype('float32') / 255.
        print(train_cell_patch.shape)

        train_label = np.concatenate(train_label_list, axis=0)

        # get intermediate feature layer
        # cell_patches = output_obj['cell_patches']
        # labels = output_obj['labels']

        train_features = self.get_features(train_cell_patch)
        features = self.get_features(self.cell_patches)
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=1223)
        clf = svm.SVC(kernel='linear')

        y_pred_all = []
        for random_state in opt_random_state:
            print("random seed:", random_state)
            # _, train_features_subset, _, train_label_subset = train_test_split(train_features, train_label, test_size=opt_test_size, random_state=opt_random_state)
            train_features_subset, train_label_subset = self.get_svm_train_subset(train_features, train_label,
                                                                                  opt_test_size, cell_classes,
                                                                                  random_state)

            print(np.unique(train_label_subset, return_counts=True))
            print("No. of training samples:", train_label_subset.shape[0])

            clf.fit(train_features_subset, train_label_subset)
            y_pred = clf.predict(features)
            y_pred_all.append(y_pred)
            cm = confusion_matrix(self.labels, y_pred)
            print(cm)
            print(classification_report(self.labels, y_pred, digits=4))

        return y_pred_all