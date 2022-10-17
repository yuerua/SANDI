# Description: Extract single cell images from tiles
import os
import numpy as np
# from glob import glob

def get_im_batch(im, csv, patch_size, patch_obj):
    cell_list = []
    label_list=[]
    for i, row in csv.iterrows():
        label = row['V1']
        x = row['V2']
        y = row['V3']
        pad = np.int(np.ceil(patch_size / 2))
        im_pad = np.lib.pad(im, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
        x0 = x
        x1 = x + pad * 2
        y0 = y
        y1 = y + pad * 2
        cell_patch = im_pad[y0:y1, x0:x1, :]
        split_patch = patch_obj.extract_patches(cell_patch)
        cell_list.append(split_patch)
        label_list.append(label)
    # save_f = os.path.join(save_dir, im_name+'_'+str(i)+'.npy')
    # sio.imsave(save_f,cell_patch)
    cell_list = np.array(cell_list)
    label_list = np.array(label_list)
    return cell_list, label_list

def get_slide_batch(im_fs, cell_classes):
    cell_patches = []
    labels = []
    im_idex = []
    im_names = []

    for im_f in im_fs:
        im_name = os.path.splitext(os.path.basename(im_f))[0]
        im_cell_patches, im_labels = np.load(im_f, allow_pickle=True)
        im_idex_count = 0
        for i in range(im_cell_patches.shape[0]):
            if str(im_labels[i]) in cell_classes or not cell_classes:
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

