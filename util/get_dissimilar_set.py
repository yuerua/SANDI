# Description: find the cell image with the largest distance to the reference
import numpy as np
import os
import skimage.io as sio

def get_dissimilar_set(pre_labels, pre_distance, cell_patches, labels, cell_classes,
                       save_path, patch_obj, patch_size, select_no=1, diff_only = False, save_ref = True):
    add_ref_patches = []
    add_ref_labels = []
    add_idx = []
    for c in cell_classes:
        dist_c = pre_distance[np.where(pre_labels==c)[0]]
        #sort and get the nth maximum
        if len(dist_c) < select_no:
            print('No. of predicted %s smaller than given no, adding %s patches to ref'%(c, str(len(dist_c))))
            max_i = np.where(pre_labels == c)[0][np.argsort(dist_c)[-len(dist_c):]]
            # max_i = np.where(pre_labels == c)[0][np.argsort(dist_c)[:len(dist_c)]]
        else:
            max_i = np.where(pre_labels==c)[0][np.argsort(dist_c)[-select_no:]]
            # max_i = np.where(pre_labels == c)[0][np.argsort(dist_c)[:select_no]]

        #keep only difference
        if diff_only:
            for m_i in max_i:
                if labels[m_i] != c:
                    add_ref_patches.append(cell_patches[m_i])
                    add_ref_labels.append(labels[m_i])
                    add_idx.append(m_i)
        else:
            add_ref_patches.append(cell_patches[max_i])
            add_ref_labels.append(labels[max_i])
            add_idx.append(max_i)

    add_ref_patches = np.concatenate(add_ref_patches,0)
    add_ref_labels = np.concatenate(add_ref_labels, 0)
    add_idx = np.concatenate(add_idx, 0)
    print('Additional ref labels', np.unique(add_ref_labels, return_counts=True))

    if save_ref:
        for add_i in add_idx:
            add_im = patch_obj.merge_patches(np.zeros((patch_size, patch_size,3)), cell_patches[add_i])
            add_im = (add_im*255.).astype('uint8')
            #i_pred_gt.png
            save_add_f = os.path.join(save_path, "ref_%i_%s_%s.png"%(add_i, pre_labels[add_i], labels[add_i]))
            sio.imsave(save_add_f, add_im)

    return add_ref_patches, add_ref_labels