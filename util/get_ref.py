# Created by hzhang at 19/04/2021
# Description:
import random
import numpy as np
from glob import glob
import os
import skimage.io as sio
import re

def get_ref(cell_classes, patch_obj, ref_img_path):
    # Ref patches [cell appearance has to be really typical!]
    ref_patches = []
    ref_labels = []
    # cell_classes = np.unique(labels).tolist()

    # if select_random_ref:
    #
    #     for c_label in cell_classes:
    #         min_no = np.min(np.unique(labels, return_counts=True)[1])
    #         if ref_no and ref_no <= min_no:
    #             print('Randomly select %i reference images for each class' % ref_no)
    #             c_patches = cell_patches[random.sample(np.where(labels == c_label)[0].tolist(), ref_no)]
    #         else:
    #             print('Randomly select %i reference images for each class' % min_no)
    #             c_patches = cell_patches[random.sample(np.where(labels == c_label)[0].tolist(), min_no)]
    #         ref_patches.append(c_patches)
    #         ref_labels += [c_label]
    #
    # elif ref_img_path:
    ref_im_fs = sorted([f for f in os.listdir(ref_img_path) if f.endswith('.png')])

    for ref_im_f in ref_im_fs:
        ref_im_name = os.path.splitext(ref_im_f)[0]
        ref_im_label = re.search('([^_]+$)', ref_im_name).group()

        #ref_im_label = ref_im_label.replace(":","/")
        
        # assert ref_im_label in cell_classes, \
        #     "Please specify correct cell class in ref filenames, eg. Da100_2_cd4.png"
        if ref_im_label in cell_classes or not cell_classes:
            ref_im = sio.imread(os.path.join(ref_img_path, ref_im_f))
            ref_im_patch = patch_obj.extract_patches(ref_im)
            # Caution! Due to padding in merge_patches, ref_im_f[i_r] != cell_patches[i_ori], so have to find the
            # matching index in cell_patches from ref_im filename.
            # ref_im_idx = int(re.search('(?<=_)[^_]+(?=_)', ref_im_name).group())
            # ref_im_patch = cell_patches[ref_im_idx]
            ref_patches.append(ref_im_patch)
            ref_labels.append(ref_im_label)

    assert ref_labels, \
        "Please specify correct cell class in ref filenames, eg. Da100_2_cd4.png"

    zip_ref = sorted(zip(ref_patches, ref_labels), key=lambda x: x[1])
    ref_patches, ref_labels = zip(*zip_ref)

    # ref_0 = patch_obj.merge_patches(np.zeros((28,28,3)), ref_patches[10])
    # plt.imshow(ref_0)

    ref_patches = np.stack(ref_patches, 0)
    ref_labels = np.array(ref_labels)
    if np.max(ref_patches) > 1:
        ref_patches = ref_patches.astype('float32') / 255.
    print('Reference labels:', np.unique(ref_labels, return_counts=True))

    return ref_patches, ref_labels


def get_output_ref(cell_patches, labels, ref_patches, ref_labels, patch_no, input_shape, model):
    patch_pairs = [[x, y] for x in range(patch_no) for y in range(patch_no)]
    patch_pairs = np.transpose(np.array(patch_pairs)).tolist()

    pre_labels = []
    pre_score = []
    pre_batch_score = []
    for i in range(cell_patches.shape[0]):
        im_1 = cell_patches[i]  # 9*20*20
        cell_score = []
        # if compare_max:
        for ref_i in range(ref_patches.shape[0]):
            im_0 = ref_patches[ref_i]
            # im_0 = np.reshape(im_0, (-1, input_shape[0], input_shape[1], input_shape[2]))

            pair_batch_0 = im_0[patch_pairs[0]]
            pair_batch_1 = im_1[patch_pairs[1]]

            batch_score_s = model.predict([pair_batch_0, pair_batch_1])[1]
            batch_score_r = model.predict([pair_batch_1, pair_batch_0])[1]

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