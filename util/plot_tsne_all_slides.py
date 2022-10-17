import os
import matplotlib.pyplot as plt
import matplotlib
import platform
import seaborn as sns
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

    cell_classes = np.unique(labels)
    colors = plt.get_cmap('twilight')(np.linspace(0, 1, len(cell_classes)))

    f = plt.figure(figsize=(10, 8))
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
    return f

if __name__ == "__main__":

    input_dir = "../ExpDir/ova_t/auto_ref_1/r9"
    save_path = "../plots/ova_t"
    slide_ext = ".ndpi"

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

