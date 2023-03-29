import os 
import os.path as osp 
import argparse
from easydict import EasyDict as edict

from utils import define

_C = edict()

# common 
_C.seed = 42
_C.num_workers = 4

# path params
_C.data_dir = define.SCAN3R_ORIG_DIR
_C.label_file_name = define.LABEL_FILE_NAME_GT
_C.predicted_sg = False

# preprocess params
_C.preprocess = edict()
_C.preprocess.pc_resolutions = [512] # [32, 64, 128, 256, 512]

# Data params
_C.data = edict()
_C.data.subscenes_per_scene = 7
_C.data.filter_segment_size = 512
_C.data.min_obj_points = 50
_C.data.name = '3RScan'

# Training params
_C.train = edict()
_C.train.batch_size = 4
_C.train.num_workers = 4
_C.train.pc_res = 512
_C.train.use_augmentation = True
_C.train.rot_factor = 1.0
_C.train.augmentation_noise = 0.005
_C.train.modules = ['gat', 'point', 'rel', 'attr']
_C.train.end_epoch = 50
_C.train.learning_rate = 1e-3

# Validation params
_C.val = edict()
_C.val.data_mode = 'orig'

# model param
_C.model = edict()
_C.model.rel_dim = 41
_C.model.attr_dim = 164
_C.model.zoom = 0.1
_C.model.alignment_thresh = 0.4

# registration model params
_C.reg_model = edict()
_C.reg_model.neighbor_limits = [38, 36, 36, 38]
_C.reg_model.num_p2p_corrs = 20000
_C.reg_model.corr_score_thresh = 0.1
_C.reg_model.rmse_thresh = 0.2 
_C.reg_model.inlier_ratio_thresh = 0.05

def make_cfg():
    return _C

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    args = parser.parse_args()
    return args

def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, 'output')




