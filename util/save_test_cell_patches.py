import skimage.io as sio
import os
import pandas as pd
import numpy as np

def save_cell_patches(save_dir, cws_path, csv_path, slide_name, im_name_list, patch_size, pre_labels):
    print('Processing', slide_name)
    im_name_list_1 = list(sorted(set(im_name_list), key=im_name_list.index))
    cell_idx = 0
    labels_slide = []
    for im_i in range(len(im_name_list_1)):
        im_name_slide = im_name_list_1[im_i]
        csv_f_slide = os.path.join(csv_path, slide_name, im_name_slide + '.csv')
        print('Saving cell patches from %s' % im_name_slide)
        csv_slide = pd.read_csv(csv_f_slide)

        im_f_slide = os.path.join(cws_path, slide_name, im_name_slide + '.jpg')
        im_slide = sio.imread(im_f_slide)

        for i, row in csv_slide.iterrows():
            label_im = row['V1']
            pre_label_im = pre_labels[cell_idx]
            x = row['V2']
            y = row['V3']
            pad = np.int(np.ceil(patch_size / 2))
            im_pad = np.lib.pad(im_slide, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')
            x0 = x
            x1 = x + pad * 2
            y0 = y
            y1 = y + pad * 2
            cell_patch = im_pad[y0:y1, x0:x1, :]
            cell_patch = cell_patch.astype('uint8')
            labels_slide.append(label_im)
            #Have to save as png!
            save_f = os.path.join(save_dir, im_name_slide + '_%i_%i_%i_%s_%s.png'% \
                                  (cell_idx,x,y,label_im, pre_label_im))
            sio.imsave(save_f, cell_patch)
            cell_idx += 1

def save_cell_patches_from_npy(cell_patches, im_name_list, pre_labels, labels, patch_obj, patch_size, save_dir):
    for i in range(cell_patches.shape[0]):
        merge_img = patch_obj.merge_patches(np.zeros((patch_size, patch_size, 3)), cell_patches[i])
        merge_img = merge_img*255.
        merge_img = merge_img.astype('uint8')
        sio.imsave(os.path.join(save_dir, im_name_list[i] + '_%s_%s_%s.png'%(str(i), pre_labels[i], labels[i])), merge_img)
    print('Test img saved to', save_dir)

def save_cell_patches_predict(cell_patches, slide_name_f, im_name, pre_labels, patch_obj, patch_size, save_path):
    for i in range(cell_patches.shape[0]):
        merge_img = patch_obj.merge_patches(np.zeros((patch_size, patch_size, 3)), cell_patches[i])
        merge_img = merge_img*255.
        merge_img = merge_img.astype('uint8')
        save_patch_dir = os.path.join(save_path, slide_name_f, im_name)
        if os.path.exists(save_patch_dir) is False:
            os.makedirs(save_patch_dir)
        sio.imsave(os.path.join(save_patch_dir, '%s_%s.png'%(str(i), pre_labels[i])), merge_img)
    print('Cell patch img saved to', save_patch_dir)
