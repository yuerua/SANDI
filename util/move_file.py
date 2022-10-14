# Created by hzhang at 16/04/2021
# Description: functions for sorting files
import os
from glob import glob
import random
import numpy as np


def train_valid_test_split(dataset, train_valid_split_ratio=0.2, test_split=0.15):
    test_data = []
    idx_test = random.sample(range(0, len(dataset)), int(np.ceil(len(dataset) * test_split)))
    for i in idx_test:
        test_data.append(dataset[i])
    train_valid_data = list(set(dataset) - set(test_data))
    valid_data = []
    idx = random.sample(range(0, len(train_valid_data)), int(np.ceil(len(train_valid_data) * train_valid_split_ratio)))
    for i in idx:
        valid_data.append(train_valid_data[i])
    train_data = list(set(train_valid_data) - set(valid_data))

    print('Train data number: ', len(train_data))
    print('Valid data number: ', len(valid_data))
    print('Test data number: ', len(test_data))
    return train_data, valid_data, test_data

def test_split(dataset, test_split=0.15):
    test_data = []
    idx_test = random.sample(range(0, len(dataset)), int(np.ceil(len(dataset) * test_split)))
    for i in idx_test:
        test_data.append(dataset[i])
    train_valid_data = list(set(dataset) - set(test_data))

    print('Test data number: ', len(test_data))
    return train_valid_data, test_data

def remove_model(model_dir, model_name):
    model_all = glob(os.path.join(model_dir, model_name, "*.h5"))
    model_all_idx = [int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]) for f in model_all]
    max_iter_i = model_all_idx.index(max(model_all_idx))
    del model_all[max_iter_i]
    for f in model_all:
        os.remove(f)
