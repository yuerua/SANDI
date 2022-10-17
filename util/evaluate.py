# Description: Evaluating functions

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import platform
if platform.system() == 'Darwin':
    matplotlib.use('MacOSX')

def evaluate(labels, pre_labels):
    correct_no = 0
    incorrect_idx = []
    for i in range(len(pre_labels)):
        if pre_labels[i] == labels[i]:
            correct_no += 1
        else:
            incorrect_idx.append(i)

    acc = correct_no / len(pre_labels)
    print('Accuracy:', acc)

    # confusion matrix
    cr_m = classification_report(labels, pre_labels, digits=4,  output_dict=True)
    print(classification_report(labels, pre_labels, digits=4))
    cr_m_df = pd.DataFrame(cr_m).transpose()

    cm = confusion_matrix(labels, pre_labels)
    cell_classes_slide = sorted(np.unique(np.concatenate((labels, pre_labels), axis=0)))
    df_by_class = report_cm_by_class(cm, cell_classes_slide)
    df_cm = pd.DataFrame(cm, index=[i for i in cell_classes_slide],
                         columns=[i for i in cell_classes_slide])
    print(df_cm)
    sns.set(rc={'figure.figsize':(11.0,8.27)}, font_scale=1.4)
    #cmap=sn.cubehelix_palette(light=0.9, as_cmap=True)
    hm = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d')  # font size

    return df_by_class, cr_m_df, df_cm, hm, incorrect_idx

def report_cm_by_class(cm, cell_classes):
    df_by_class = pd.DataFrame(columns=["class","TP","TN","FP","FN","sensitivity","specificity","precision","f1_score","support"])

    for c in range(cm.shape[0]):
        TP = cm[c,c]
        FP = np.sum(cm[:,c]) - TP
        FN = np.sum(cm[c,:]) - TP
        TN = np.sum(cm) - FP - FN - TP
        support = np.sum(cm[c,:])

        sensitivity = TP/(TP+FN)
        precision = TP/(TP+FP)
        specificity = TN / (TN + FP)
        f1_score = 2 * (sensitivity * precision) / (sensitivity + precision)

        df_by_class.loc[c] = [cell_classes[c], TP, TN, FP, FN, sensitivity, specificity, precision, f1_score, support]

    return df_by_class


def test_cell_pairs(im_0, im_1, model, patch_obj, patch_no, loss_type):
    if np.max(im_1) > 1.:
        im_1 = im_1.astype('float32') / 255.
    if np.max(im_0) > 1.:
        im_0 = im_0.astype('float32') / 255.
    if len(im_1.shape) == 3:
        im_1 = patch_obj.extract_patches(im_1)
    if len(im_0.shape) == 3:
        im_0 = patch_obj.extract_patches(im_0)

    patch_pairs = [[x, y] for x in range(patch_no) for y in range(patch_no)]
    patch_pairs = np.transpose(np.array(patch_pairs)).tolist()

    pair_batch_0 = im_0[patch_pairs[0]]
    pair_batch_1 = im_1[patch_pairs[1]]

    # pair_batch_0 = np.expand_dims(im_0[4], axis=0)
    # pair_batch_1 = np.expand_dims(im_1[4], axis=0)

    if loss_type == "bce_only":
        batch_score_s = model.predict([pair_batch_0, pair_batch_1])
        batch_score_r = model.predict([pair_batch_1, pair_batch_0])
    else:
        batch_score_s = model.predict([pair_batch_0, pair_batch_1])[1]
        batch_score_r = model.predict([pair_batch_1, pair_batch_0])[1]

    batch_score = (batch_score_s + batch_score_r) / 2.

    # wrong_im_0 = pair_batch_0[np.where(batch_score == np.max(batch_score))[0]]
    # wrong_im_1 = pair_batch_1[np.where(batch_score == np.max(batch_score))[0]]
    # wrong_im_0 = np.reshape(wrong_im_0, input_shape)
    # wrong_im_1 = np.reshape(wrong_im_1, input_shape)
    #plt.imshow(wrong_im_0)
    #plt.imshow(wrong_im_1)
    his = plt.hist(batch_score, bins=50)
    #print(batch_score)
    return batch_score



