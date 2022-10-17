import os
from glob import glob
# import pandas as pd
from skimage import io as sio
import numpy as np
from util.Patches import Patches

def npy_to_im(npy_dir, slide_ext, cell_img_save_dir, patch_obj, patch_size, cell_classes):
    slides = sorted(glob(os.path.join(npy_dir, "*" + slide_ext)))
    for slide in slides:
        cell_idx = 0
        slide_name = os.path.basename(slide)

        if os.path.exists(os.path.join(cell_img_save_dir, slide_name)) is False:
            os.makedirs(os.path.join(cell_img_save_dir, slide_name))

        print('npy to image:', slide_name)
        npy_fs = sorted(glob(os.path.join(slide, "Da*.npy")))
        for npy_f in npy_fs:
            im_name = os.path.splitext(os.path.basename(npy_f))[0]

            #Save cell patch img
            data_all = np.load(npy_f, allow_pickle=True)
            #data[0]:img, data[1]:label
            # if len(data_all.shape) ==2:
            data = data_all[0]
            labels = data_all[1]
            # csv_f = os.path.join(csv_dir, slide_name, im_name + '.csv')
            # csv = pd.read_csv(csv_f)

            for i in range(data.shape[0]):
                label = labels[i]
                if str(label) in cell_classes or not cell_classes:
                    cell_patch = data[i]
                    if np.max(cell_patch) <= 1.:
                        cell_patch = cell_patch * 255.

                # for i, row in csv.iterrows():
                #     label = row['V1']
                #     cell_patch = data[i]
                #     if np.max(cell_patch) <= 1.:
                #         cell_patch = cell_patch * 255.

                    cell_patch = patch_obj.merge_patches(np.zeros((patch_size, patch_size, 3)), cell_patch)
                    cell_patch = cell_patch.astype('uint8')
                    save_f = os.path.join(cell_img_save_dir, slide_name, im_name + '_' + str(cell_idx) + '_' + str(label) + '.png')
                    sio.imsave(save_f, cell_patch)
                    cell_idx += 1

if __name__ == '__main__':
    # cws_path = '/Volumes/proj4/Gal8/data/cws'
    npy_dir = '/Users/hzhang/Documents/project/siamese/formal_1/hypoxia/data/data_20_4_with_none/ref_img'
    csv_dir = '/Users/hzhang/Documents/project/siamese/formal_1/hypoxia/Misc/celllabels'
    cell_img_save_dir = '/Users/hzhang/Documents/project/siamese/formal_1/hypoxia/ref_img/cell_img'
    patch_size = 28
    sub_patch_size = 20
    sub_patch_stride = 4

    patch_obj = Patches(
        img_patch_h=sub_patch_size, img_patch_w=sub_patch_size,
        stride_h=sub_patch_stride, stride_w=sub_patch_stride,
        label_patch_h=sub_patch_size, label_patch_w=sub_patch_size)

