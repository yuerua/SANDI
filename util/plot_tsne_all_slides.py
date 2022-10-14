import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import platform
import seaborn as sns
import cv2
from sklearn.manifold import TSNE
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')
import numpy as np


def tsne_represent(features, labels, n_iter=500):
    sns.set_style(style='white')
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
    cell_classes = np.unique(labels)
    colors = plt.get_cmap('twilight')(np.linspace(0, 1, len(cell_classes)))

    f = plt.figure(figsize=(10, 8))
    # ax = f.add_subplot(1, 1)
    for idex, (c_color, c_label) in enumerate(zip(colors, cell_classes)):
        plt.scatter(tsne_features[np.where(labels == c_label), 0],
                    tsne_features[np.where(labels == c_label), 1],
                    marker='o',
                    #color=c_color,
                    linewidth='1',
                    alpha=0.8,
                    label=c_label)

    # ax.set_axis_bgcolor('white')
    sns.set_style(style='white')
    plt.rcParams['figure.facecolor'] = 'white'
    # plt.axes.set_facecolor("white")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on Testing Samples')
    plt.legend(loc='best')
    #plt.pause(5)

    return f

if __name__ == "__main__":

    #Ovarian PDL1
    # input_dir = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/ova_pdl1_050622_best/results/pdl1_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_0098/auto_ref_1/r9"
    # save_path = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/ova_pdl1_050622_best/results/pdl1_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_0098/"
    # slide_ext = ".ndpi"

    #IMC_CD4_CD8
    # input_dir = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/IMC_immune_CD4_CD8/results/IMC_immune_CD4_CD8_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_097/auto_ref/r9"
    # save_path = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/IMC_immune_CD4_CD8/results/IMC_immune_CD4_CD8_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_097/"
    # slide_ext = ".ndpi"

    #Ova T
    # input_dir = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/ova_t/results/ova_t_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_0100/auto_ref_best/r9"
    # save_path = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/ova_t/results/ova_t_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_0100/"
    # slide_ext = ".ndpi"

    #LUSC T
    # input_dir = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/lusc_t/results/lusc_t_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_090_new/auto_ref_3/r9"
    # save_path = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/lusc_t/results/lusc_t_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_090_new/"
    # slide_ext = ".ndpi"

    #hypoxia_cap_can
    # input_dir = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/hypoxia_cap_can/results/hypoxia_cap_can_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_059/auto_ref_3/r9"
    # save_path = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/hypoxia_cap_can/results/hypoxia_cap_can_unsuper_b_256_opt_adam_combined_ratio_0.7_0.3_059/"
    # slide_ext = ".svs"

    #hypoxia_fop_fon
    # input_dir = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/hypoxia_fop_fon/results/hypoxia_fop_fon_unsuper_b_256_opt_adam_contrast_only_ratio_0.7_0.3_094/auto_ref_3/r9"
    # save_path = "/Users/hzhang/Documents/project/SANDI/SANDI/ExpDir/hypoxia_fop_fon/results/hypoxia_fop_fon_unsuper_b_256_opt_adam_contrast_only_ratio_0.7_0.3_094/"
    # slide_ext = ".svs"


    slide_list = [i for i in os.listdir(input_dir) if i.endswith(slide_ext)]
    feature_all = []
    label_all = []
    for slide in slide_list:
        slide_feature = np.load(os.path.join(input_dir, slide, "features.npy"), allow_pickle=True)
        slide_output = np.load(os.path.join(input_dir, slide, "output.npy"), allow_pickle=True)
        slide_label = slide_output.take(0)['labels']
        label_all.append(slide_label)
        feature_all.append(slide_feature)

    label_all = np.concatenate(label_all, axis=0)
    feature_all = np.concatenate(feature_all, axis=0)

    tsne_fig = tsne_represent(feature_all, label_all, n_iter=500)
    tsne_fig.savefig(os.path.join(save_path, 'tsne_all.pdf'))

