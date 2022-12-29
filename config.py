'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''
#TM_BL_T1_CT_OneClassLearning_Dataset_220620
# device settings
device = 'cuda'  # or 'cpu'

# data settings
dataset_path = "/home/edshkim98/synapse/one_class/" #"/home/edshkim98/synapse/one_class/"  # parent directory of datasets
class_name = "SMT_cut"  #TM_FP_T3_DT_OneClassLearning_Dataset_220620 dataset subdirectory
modelname = "synapse_smt"  # export evaluations/logs with this name
pre_extracted = False  # were feature preextracted with extract_features?
finetune= True

img_size = (512, 512)  # image size of highest scale, others are //2, //4
assert img_size[0] % 128 == 0 and img_size[1] % 128 == 0, "image width/height should be a multiple of 128"

img_dims = [3] + list(img_size)

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3  # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp = 3  # clamping parameter
max_grad_norm = 1e0  # clamp gradients to this norm
n_coupling_blocks = 4  # higher = more flexible = more unstable
fc_internal = 256 * 4 # number of neurons in hidden layers of s-t-networks
lr_init = 0.001  # inital learning rate 0.001
use_gamma = True

extractor = "effb5"  # feature dataset name (which was used in 'extract_features.py' as 'export_name')
n_feat = [64,176,304] #{"effnetB5": 304}[extractor]  # dependend from feature extractor 176 256 1024
div = 4
map_size = [(img_size[0] // div, img_size[1] // div), (img_size[0] // div //2, img_size[1] // div//2), (img_size[0] // div //4, img_size[1] // div //4)]

# dataloader parameters
batch_size = 16  # actual batch size is this value multiplied by n_transforms(_test)
kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 5  # total epochs = meta_epochs * sub_epochs
sub_epochs = 100  # evaluate after this number of epochs

# output settings
verbose = True
hide_tqdm_bar = True
save_model = True