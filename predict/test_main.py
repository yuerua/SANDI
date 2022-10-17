import os
from glob import glob
import numpy as np
import skimage.io as sio
import pandas as pd
import math
from shutil import copyfile, rmtree
import time
import matplotlib.pyplot as plt
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')

import random
from train.model import get_similarity_model
from util.plot import tsne_represent, put_markers
from util.get_batch import get_im_batch, get_slide_batch
from util.get_ref import get_ref, get_output_ref
from util.get_output import Output
from util.save_test_cell_patches import save_cell_patches_from_npy
from util.evaluate import evaluate
from util.Patches import Patches
from util.get_dissimilar_set import get_dissimilar_set
random.seed(655)

class Test(object):
    def __init__(self,
        test_npy_path = '../ExpDir/ova_t/data/test/',
        ref_img_path = "../ExpDir/ova_t/ref_img/",
        results_dir = '../results/ova_t/auto_r',
        slide_ext = '.svs',
        # Network settings
        model_dir = '../ExpDir/ova_t/model/',
        model_name = 'ova_t_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_0100.h5',
        loss_type = "combined",
        # patch settings
        patch_size = 28,
        sub_patch_size = 20,
        sub_patch_stride = 4,
        cell_classes=sorted(['can', 'cap']),

        mode="SVM",
        get_tsne=True,
        get_SVM=False,
        opt_random_state=[98, 1006, 665, 379, 507],
        SVM_train_npy_path="",
        opt_test_size=0.002,
        save_test_img = False, #convert npy to cell patch imgs?
        save_pred_patch = False,
        color_code_f='../ExpDir/ova_t/ovarian_color.csv',

        automatic_ref_npy_path = "",
        automatic_rounds=0,
        **kwargs):


        self.test_npy_path = test_npy_path
        self.ref_img_path = ref_img_path
        self.slide_ext = slide_ext

        self.model_name = model_name
        if self.model_name.endswith('.h5'):
            self.model_name = self.model_name[0:-3]
        self.model_weight_dir = os.path.join(model_dir, self.model_name)

        self.loss_type = loss_type

        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride
        self.cell_classes = cell_classes
        self.input_shape = (self.sub_patch_size, self.sub_patch_size, 3)
        self.patch_no = int((math.ceil((self.patch_size - self.sub_patch_size) / self.sub_patch_stride) + 1) ** 2)

        self.mode = mode
        self.get_tsne = get_tsne
        self.get_SVM = get_SVM
        self.save_test_img = save_test_img

        self.opt_random_state = opt_random_state
        self.opt_test_size = opt_test_size
        self.SVM_train_npy_path = SVM_train_npy_path

        self.save_pred_patch = save_pred_patch
        self.color_code_f = color_code_f

        #Get model
        self.model = get_similarity_model(self.input_shape, self.loss_type)

        model_set = [f for f in os.listdir(self.model_weight_dir) if f.endswith(".h5")]
        model_set_i = [int(os.path.splitext(f)[0].split("_")[-1]) for f in model_set]
        best_model = model_set[model_set_i.index(max(model_set_i))]
        print(best_model)

        self.save_path = os.path.join(results_dir, best_model[0:-3])

        if os.path.exists(self.save_path) is False:
            os.makedirs(self.save_path)

        self.model.load_weights(os.path.join(self.model_weight_dir, best_model))

        self.patch_obj = Patches(
        img_patch_h=self.sub_patch_size, img_patch_w=self.sub_patch_size,
        stride_h=self.sub_patch_stride, stride_w=self.sub_patch_stride,
        label_patch_h=self.sub_patch_size, label_patch_w=self.sub_patch_size)

        self.automatic_ref_npy_path = automatic_ref_npy_path
        self.automatic_rounds = automatic_rounds

        if self.automatic_ref_npy_path and self.automatic_rounds >0:
            self.get_automatic_ref()

        # generate ref dataset
        print(self.ref_img_path)
        self.ref_patches, self.ref_labels = get_ref(self.cell_classes, self.patch_obj, self.ref_img_path)


    def run(self):
        print('Number of patches per cell:', self.patch_no)

        test_slide = sorted(glob(os.path.join(self.test_npy_path, '*' + self.slide_ext)))
        assert test_slide, "Testing samples not found"

        pre_labels_all = []
        labels_all = []

        for slide in test_slide:
            slide_name = os.path.basename(slide)

            im_fs = sorted(glob(os.path.join(slide, '*.npy')))

            cell_patches, labels, im_name_list = get_slide_batch(im_fs, self.cell_classes)
            labels_all.append(labels)

            print('Total number of cells', len(im_name_list))

            output_slide = Output(cell_patches=cell_patches,labels = labels, model=self.model,
                                  ref_patches=self.ref_patches, ref_labels=self.ref_labels,
                                  patch_size=self.patch_size, sub_patch_size=self.sub_patch_size,
                                  sub_patch_stride=self.sub_patch_stride,cell_classes=self.cell_classes, mode=self.mode,
                                  loss_type=self.loss_type)
            
            if os.path.exists(os.path.join(self.save_path, slide_name)) is False:
                os.makedirs(os.path.join(self.save_path, slide_name))

            if self.get_tsne:
                features = output_slide.get_features(cell_patches)
                tsne_fig = tsne_represent(features, labels, n_iter=500)
                tsne_fig.savefig(os.path.join(self.save_path, slide_name, 'tsne.png'))
                np.save(os.path.join(self.save_path, slide_name, 'features.npy'), features)
                plt.clf()

            if self.get_SVM:
                SVM_pred_all = output_slide.SVM_evaluate(self.SVM_train_npy_path, self.slide_ext, self.opt_random_state,
                                             self.cell_classes, self.opt_test_size)
                np.save(os.path.join(self.save_path, slide_name, "_%.2f_SVM_pred_all.npy"%self.opt_test_size),
                        SVM_pred_all, allow_pickle=True)

            #predict
            print('Processing test set for', slide_name)

            pre_labels, pre_score, pre_batch_score, pre_distance = output_slide.output(self.mode)
            pre_labels_all.append(pre_labels)

            #evaluate
            print('Evaluating', slide_name)
            df_by_class, cr_m_df, df_cm, hm, incorrect_idx = evaluate(labels, pre_labels)
            #save
            df_by_class.to_csv(os.path.join(self.save_path, slide_name, 'evaluate_by_class.csv'), encoding='utf-8')
            cr_m_df.to_csv(os.path.join(self.save_path, slide_name, 'evaluate_slide.csv'), encoding='utf-8')
            df_cm.to_csv(os.path.join(self.save_path, slide_name, 'cm.csv'), encoding='utf-8')
            hm.figure.savefig(os.path.join(self.save_path, slide_name, 'cm.pdf'))
            plt.clf()

            output_obj = {'cell_patches': cell_patches, 'ref_labels':self.ref_labels,'labels':labels,
                          'pre_labels':pre_labels, 'pre_score': pre_score, 'incorrect_idx':incorrect_idx,
                          'pre_batch_score': pre_batch_score, 'pre_distance':pre_distance}
            np.save(os.path.join(self.save_path, slide_name, 'output.npy'),output_obj, allow_pickle=True)

            if self.save_test_img:
                save_dir = os.path.join(self.save_path, slide_name, 'test_img')
                if os.path.exists(save_dir) is False:
                    os.makedirs(save_dir)
                # saved_labels = save_cell_patches(save_dir, slide_name, im_name_list, patch_size, pre_labels)
                # assert (saved_labels==labels).all(), "Image order doesn't match with npy"
                if self.save_test_img=="incorrect_only":
                    save_cell_patches_from_npy(np.array(cell_patches)[incorrect_idx], np.array(im_name_list)[incorrect_idx],
                                               np.array(pre_labels)[incorrect_idx], np.array(labels)[incorrect_idx],
                                               self.patch_obj, self.patch_size, save_dir)
                else:
                    save_cell_patches_from_npy(cell_patches, im_name_list, pre_labels, labels,
                                               self.patch_obj, self.patch_size, save_dir)

        print('Evaluating all slides')
        labels_all = np.concatenate(labels_all, axis=0)
        pre_labels_all = np.concatenate(pre_labels_all, axis=0)
        df_by_class, cr_m_df, df_cm, hm, incorrect_idx = evaluate(labels_all, pre_labels_all)
        # save
        df_by_class.to_csv(os.path.join(self.save_path, 'evaluate_by_class_all.csv'), encoding='utf-8')
        cr_m_df.to_csv(os.path.join(self.save_path, 'evaluate_slide_all.csv'), encoding='utf-8')
        df_cm.to_csv(os.path.join(self.save_path, 'cm_all.csv'), encoding='utf-8')
        hm.figure.savefig(os.path.join(self.save_path, 'cm_all.png'))

    def get_automatic_ref(self):

        ref_npy_pool = sorted(glob(os.path.join(self.automatic_ref_npy_path, "**", "*.npy")))
        assert ref_npy_pool, "No npy found for selecting automatic references"
        cell_patches, labels, im_name_list = get_slide_batch(ref_npy_pool, self.cell_classes)

        ref_img_path_auto = os.path.join(self.save_path, "auto_ref")
        if os.path.exists(ref_img_path_auto):
            rmtree(ref_img_path_auto)
        os.makedirs(ref_img_path_auto)

        ref_img_init = glob(os.path.join(self.ref_img_path, "*.png"))
        for img_f in ref_img_init:
            copyfile(img_f, os.path.join(ref_img_path_auto, os.path.basename(img_f)))

        for i in range(self.automatic_rounds):
            print("Auto rounds %i"%i)
            ref_patches, ref_labels = get_ref(self.cell_classes, self.patch_obj, ref_img_path_auto)

            output_auto_ref = Output(cell_patches=cell_patches,labels = labels, model=self.model,
                                  ref_patches=ref_patches, ref_labels=ref_labels,
                                  patch_size=self.patch_size, sub_patch_size=self.sub_patch_size,
                                  sub_patch_stride=self.sub_patch_stride,cell_classes=self.cell_classes,
                                  mode="compare_feature_dist",
                                  loss_type=self.loss_type)

            pre_labels, pre_score, pre_batch_score, pre_distance = output_auto_ref.output("compare_feature_dist")
            add_ref_patches, add_ref_labels = get_dissimilar_set(pre_labels, pre_score, cell_patches, labels,
                                                                 self.cell_classes, ref_img_path_auto, self.patch_obj,
                                                                 self.patch_size, select_no=1,
                                                                 diff_only = False, save_ref = True)

        self.ref_img_path = ref_img_path_auto