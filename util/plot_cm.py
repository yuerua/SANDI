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


def plot_training_curves_from_history(output_dir, H, num_of_epoch, model_name, loss_type):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use("seaborn-white")
    plt.figure()
    plt.plot(np.arange(0, num_of_epoch), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, num_of_epoch), H.history["val_loss"], label="val_loss")
    plt.title("Training/Validation Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, model_name + '_Loss.png'), dpi=800)
    # plt.show()

    plt.figure()
    if loss_type == "combined":
        plt.plot(np.arange(0, num_of_epoch), H.history['BCE_acc'], label="train_acc")
        plt.plot(np.arange(0, num_of_epoch), H.history['val_BCE_acc'], label="val_acc")
    else:
        plt.plot(np.arange(0, num_of_epoch), H.history['acc'], label="train_acc")
        plt.plot(np.arange(0, num_of_epoch), H.history['val_acc'], label="val_acc")
    plt.title("Training/Validation Networrk")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, model_name + '_Accuracy.png'), dpi=800)
    # plt.show()
    

def plot_training_curves_from_csv(output_dir, csv, model_name, loss_type):

    num_of_epoch = max(csv.epoch)+1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use("seaborn-white")
    plt.figure()
    plt.plot(np.arange(0, num_of_epoch), csv["loss"], label="train_loss")
    plt.plot(np.arange(0, num_of_epoch), csv["val_loss"], label="val_loss")
    plt.title("Training/Validation Network")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, model_name + '_Loss.png'), dpi=800)
    # plt.show()

    plt.figure()
    if loss_type == "combined":
        plt.plot(np.arange(0, num_of_epoch), csv['BCE_acc'], label="train_acc")
        plt.plot(np.arange(0, num_of_epoch), csv['val_BCE_acc'], label="val_acc")
    else:
        plt.plot(np.arange(0, num_of_epoch), csv['acc'], label="train_acc")
        plt.plot(np.arange(0, num_of_epoch), csv['val_acc'], label="val_acc")

    plt.title("Training/Validation Networrk")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_dir, model_name + '_Accuracy.png'), dpi=800)
    # plt.show()

def tsne_represent(features, labels, n_iter=500):
    # tsne
    sns.set_style("white")
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
    for idex, (c_color, c_label) in enumerate(zip(colors, cell_classes)):
        plt.scatter(tsne_features[np.where(labels == c_label), 0],
                    tsne_features[np.where(labels == c_label), 1],
                    marker='o',
                    #color=c_color,
                    linewidth='1',
                    alpha=0.8,
                    label=c_label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on Testing Samples')
    plt.legend(loc='best')
    #plt.pause(5)

    return f

def put_markers(csv,im, pre_labels, color_dict):
    r = 4
    for i, row in csv.iterrows():
        label_c = pre_labels[i]
        x = row['V2']
        y = row['V3']
        h = color_dict[label_c].lstrip('#')
        color_c = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        cv2.circle(im, (x, y), r, color=color_c, thickness=-1, lineType=cv2.LINE_AA)
    #        cv2.circle(image,(X[i], Y[i]), r, color=color, thickness=-1)
    return im


def plot_confusion_matrix(df_cm):
    print(df_cm)
    sns.set(rc={'figure.figsize':(11.0,8.27)}, font_scale=1.4)
    #cmap=sn.cubehelix_palette(light=0.9, as_cmap=True)
    sns.set_style(style='white')
    hm = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap="rocket")  # font size
    return hm
    
if __name__ == "__main__":

    save_path = "../plots/confusion_matrix"
    dataset_name = "myeloma"
    df_cm = pd.read_csv("../plots/confusion_matrix/myeloma/cm_all.csv", sep=",", index_col=0)

    hm = plot_confusion_matrix(df_cm)

    os.makedirs(save_path, exist_ok=True)
    hm.figure.savefig(os.path.join(save_path, dataset_name + '_cm.pdf'))