import os
from util.move_file import remove_model, test_split
import pandas as pd

from preprocess.get_cell_patch import Extractor
from util.opts import Params
from train.train import Train
from util.plot import plot_training_curves_from_csv

from predict.test_main_rounds import Test

if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Main:
    def __init__(self, opts):

        self.extract_patch = opts.extract_patch
        self.data_path = opts.data_path
        self.batch_size = opts.batch_size
        self.test_npy_path = opts.test_npy_path
        self.test_ratio = opts.test_ratio
        self.csv_path = opts.csv_path

        if self.extract_patch and os.path.exists(os.path.join(self.data_path, "train_info.npy")) is False:
            print("\n" + "=" * 20 + "Extracting data" + "=" * 20)
            extractor = Extractor(opts)
            extractor.write_to_tf_cell()
        else:
            print("\nTfrecord already exists in %s"%opts.data_path)
        #
        print("\n" + "="*20 + "Training" + "="*20)
        Trainer = Train(opts)

        # for size_ratio in [0.1, 0.2, 0.3, 1.0]:
        #     print(size_ratio)
            # H = Trainer.train_and_predict_subset_supervised_balance(size_ratio)
        Trainer.train_and_predict()

        if opts.remove_previous_model:
            print("\nKeeping the best model only")
            remove_model(opts.model_dir, opts.model_name)

        print("\nPlotting training history")
        train_hist = pd.read_csv(os.path.join(opts.model_dir, opts.model_name, "train_hist.csv"))
        plot_training_curves_from_csv(opts.history_dir, train_hist, opts.model_name, opts.loss_type)

        print("\nTesting model")

        Test(**vars(opts)).run()


if __name__ == '__main__':

    dataset_list = ['ova_t', 'lusc_t', 'dcis_fop_fon', 'myeloma', 'IMC_immune_CD4_CD8']

    cell_class_dir = {'ova_t': sorted(['cd4', 'cd8', 'foxp3', 'pd1', 'pd1cd8', 'pd1cd4']),
                      'lusc_t':sorted(["cd4", "cd8", "foxp3", "uc"]),
                      'dcis_fop_fon':sorted(["fop", "fon"]),
                      'myeloma': sorted(["CD8", "CD4", "FOXP3+CD4+"]),
                      'IMC_immune_CD4_CD8': sorted(["Th", "Tc"])}

    for dataset_name in ['ova_t']:

        if dataset_name in ['hypoxia_fop_fon']:
            slide_ext = ".svs"
        else:
            slide_ext = ".ndpi"

        for loss_type in ["bce_only", "contrast_only", "combined"]:
            weight1 = 0.7
            weight0 = 0.3

            opts = Params(
                          dataset_name=dataset_name,
                          cell_classes=cell_class_dir[dataset_name],
                          slide_ext=slide_ext,
                          loss_type = loss_type,
                          weight1=weight1,
                          weight0=weight0
                          )

            Main(opts)

        loss_type = "bce_only"
        weight1 = 1.0
        weight0 = 1.0
        opts = Params(
            dataset_name=dataset_name,
            cell_classes=cell_class_dir[dataset_name],
            slide_ext=slide_ext,
            loss_type=loss_type,
            weight1=weight1,
            weight0=weight0
        )

        Main(opts)











