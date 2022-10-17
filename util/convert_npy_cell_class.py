import numpy as np
import os
from glob import glob

def convert_cell_class(npy_fs, cell_classes_map, save_path):
    for npy_f in npy_fs:
        npy_f_name = os.path.basename(npy_f)
        print(npy_f_name)
        das = [f for f in os.listdir(npy_f) if f.endswith(".npy")]
        if os.path.exists(os.path.join(save_path, npy_f_name)) is False:
            os.makedirs(os.path.join(save_path, npy_f_name))
        
        for da in das:
            data, labels = np.load(os.path.join(npy_f, da), allow_pickle=True)
            for i in range(labels.shape[0]):
                label = str(labels[i])
                if label in cell_classes_map.keys():
                    labels[i] = cell_classes_map[label]

            save_f = os.path.join(save_path, npy_f_name, da)
            np.save(save_f, [data, labels])

if __name__ == "__main__":
    npy_fs = glob(os.path.join("../test", "*.ndpi"))
    cell_classes_map = {"pastatWeak":"pstat",
                        "pstatModerate":"pstat",
                        "pstatStrong":"pstat"}

    save_path = "../test_convert_class"

    convert_cell_class(npy_fs, cell_classes_map, save_path)