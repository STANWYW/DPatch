import os
from .config_voc import *  # noqa
from .exps.darknet19_exp1 import *  # noqa
import torch
from torch.autograd import Variable

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)


# input and output size
############################
multi_scale_inp_size = [np.array([416, 416], dtype=np.int),
                        #np.array([352, 352], dtype=np.int),
                        #np.array([384, 384], dtype=np.int),
                        #np.array([320, 320], dtype=np.int),
                        #np.array([448, 448], dtype=np.int),
                        #np.array([480, 480], dtype=np.int),
                        #np.array([512, 512], dtype=np.int),
                        #np.array([544, 544], dtype=np.int),
                        #np.array([576, 576], dtype=np.int),
                        # np.array([608, 608], dtype=np.int),
                        ]   # w, h
multi_scale_out_size = [multi_scale_inp_size[0] / 32,
                        #multi_scale_inp_size[1] / 32,
                        #multi_scale_inp_size[2] / 32,
                        #multi_scale_inp_size[3] / 32,
                        #multi_scale_inp_size[4] / 32,
                        #multi_scale_inp_size[5] / 32,
                        #multi_scale_inp_size[6] / 32,
                        #multi_scale_inp_size[7] / 32,
                        #multi_scale_inp_size[8] / 32,
                        # multi_scale_inp_size[9] / 32,
                        ]   # w, h
inp_size = np.array([416, 416], dtype=np.int)   # w, h
out_size = inp_size / 32


# for display
############################
def _to_color(indx, base):
    """ return (b, r, g) tuple"""
    base2 = base * base
    b = 2 - indx / base2
    r = 2 - (indx % base2) / base
    g = 2 - (indx % base2) % base
    return b * 127, r * 127, g * 127


base = int(np.ceil(pow(num_classes, 1. / 3)))
colors = [_to_color(x, base) for x in range(num_classes)]


# detection config
############################
thresh = 0.3


# dir config
############################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
TRAIN_DIR = os.path.join(MODEL_DIR, 'training')
TEST_DIR = os.path.join(MODEL_DIR, 'testing')

trained_model = os.path.join(MODEL_DIR, h5_fname)
pretrained_model = os.path.join(MODEL_DIR, pretrained_fname)
#pretrained_model = '/home/zwliu/FashionComp/adpatch/DPatch/yolo2-pytorch_backup/yolo-voc.weights.h5'
train_output_dir = os.path.join(TRAIN_DIR, exp_name)
print("train_output_dir %s" %train_output_dir)
test_output_dir = os.path.join(TEST_DIR, imdb_test, h5_fname)
mkdir(train_output_dir, max_depth=3)
mkdir(test_output_dir, max_depth=4)

rand_seed = 1024
use_tensorboard = True

log_interval = 50
disp_interval = 10

# properties of DPatch
target_class = 1
patch_x, patch_y = 0., 0.
patch_w, patch_h = 120., 120.
patch_size = int(patch_w-patch_x)
patch_dir = os.path.join('trained_patch', str(target_class))
img_w, img_h = 416, 416
if not os.path.exists(patch_dir):
        os.mkdir(patch_dir)
print('--------------- patch saved in ----------------')
print(patch_dir)

# for testing
patch_path = 'untargetd_DPATCH.npy'
#patch_path = "/home/zwliu/FashionComp/adpatch/DPatch/yolo2-pytorch/trained_patch/1/untargeted_DPATCH.npy"
# patch = Variable(torch.FloatTensor(np.load(patch_path)),requires_grad=False)
patch = None  # 暂时设为None，仅在测试模式需要时加载
