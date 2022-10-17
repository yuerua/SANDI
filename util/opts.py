import os

class Params:

    def __init__(self,
                 exp_dir=os.path.join(os.getcwd(), 'ExpDir'),
                 dataset_name = "try",
                 cws_path = "",
                 csv_path = "",
                 slide_ext = ".npy",
                 data_path = "",
                 ref_img_path="",
                 model_dir = "",
                 model_name="",

                 test_ratio=0.2,
                 test_npy_path="",

                 extract_patch=True,

                 train_valid_split_ratio = 0.2,
                 train_data_filename="train",
                 valid_data_filename="valid",

                 patch_size = 28,
                 sub_patch_size = 20,
                 sub_patch_stride = 4,
                 cell_classes = [],

                 pretrain_model="",
                 batch_size=256,
                 mini_batch = 5,
                 supervised=False,
                 loss_type="combined", # ["bce_only", "contrast_only", "combined"]

                 num_of_epoch=100,
                 lr=0.001,
                 optimizer_type="adam",
                 early_stopping = None,
                 weight1=0.7,
                 weight0=0.3,
                 loss_temp=0.1,
                 save_best_only=True,
                 remove_previous_model=False,

                 mode="SVM",
                 get_tsne=True,
                 get_SVM=False,
                 save_test_img=False,
                 opt_random_state=[98, 1006, 665, 379, 507],
                 SVM_train_npy_path="",
                 opt_test_size=0.002,
                 color_code_f='/Users/hzhang/Documents/project/siamese/pdl1_pred/ovarian_pdl1.csv',

                 automatic_ref_npy_path="",
                 automatic_rounds=5
                 ):

        if not exp_dir:
            self.exp_dir = os.path.join(os.getcwd(), 'ExpDir')
        else:
            self.exp_dir = exp_dir

        self.dataset_name = dataset_name
        self.exp_dir = os.path.join(self.exp_dir, self.dataset_name)
        self.cws_path = cws_path
        self.csv_path = csv_path
        self.slide_ext = slide_ext
        self.patch_size = patch_size
        self.sub_patch_size = sub_patch_size
        self.sub_patch_stride = sub_patch_stride
        self.cell_classes = sorted(cell_classes)
        self.extract_patch = extract_patch
        self.test_ratio = test_ratio

        self.train_valid_split_ratio = train_valid_split_ratio
        self.train_data_filename = train_data_filename
        self.valid_data_filename = valid_data_filename

        self.batch_size = batch_size
        self.mini_batch = mini_batch
        self.supervised = supervised
        self.loss_type = loss_type

        self.optimizer_type = optimizer_type
        self.loss_type = loss_type
        self.lr = lr
        self.weight1 = weight1
        self.weight0 = weight0
        self.loss_temp = loss_temp
        self.num_of_epoch = num_of_epoch
        self.save_best_only = save_best_only
        self.early_stopping = early_stopping

        if not model_name:
            self.model_name = self.get_model_name()
        else:
            self.model_name = model_name

        self.pretrain_model = pretrain_model
        self.remove_previous_model = remove_previous_model
        self.mode = mode
        self.save_test_img = save_test_img

        if not data_path:
            self.data_path = os.path.join(self.exp_dir, "data")
        else:
            self.data_path = data_path

        if not ref_img_path:
            self.ref_img_path = os.path.join(self.exp_dir, "ref_img")
        else:
            self.ref_img_path = ref_img_path

        self.test_npy_path = test_npy_path

        if not model_dir:
            self.model_dir = os.path.join(self.exp_dir, "model")
        else:
            self.model_dir = model_dir

        self.history_dir = os.path.join(self.exp_dir, "history")
        self.results_dir = os.path.join(self.exp_dir, "results")

        self.get_tsne = get_tsne
        self.get_SVM = get_SVM
        self.opt_random_state = opt_random_state
        self.SVM_train_npy_path = SVM_train_npy_path
        self.opt_test_size = opt_test_size
        self.color_code_f = color_code_f

        self.automatic_ref_npy_path=automatic_ref_npy_path
        self.automatic_rounds=automatic_rounds

    def get_model_name(self):
        if self.supervised:
            super = "super"
        else:
            super = "unsuper"

        model_name = "%s_%s_b_%i_opt_%s_%s_ratio_%.1f_%.1f" % \
                     (self.dataset_name, super, self.batch_size, self.optimizer_type,
                      self.loss_type, self.weight1, self.weight0)
        print("Model name:", model_name)
        return model_name


