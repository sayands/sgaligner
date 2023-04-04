import os 
import os.path as osp 
import argparse
from easydict import EasyDict as edict

from utils import define
from utils.common import ensure_dir

_C = edict()

# common 
_C.seed = 42
_C.num_workers = 4

# path params
_C.data_dir = define.SCAN3R_ORIG_DIR
_C.label_file_name = define.LABEL_FILE_NAME_GT
_C.predicted_sg = False
_C.modules = ['gat', 'point', 'rel', 'attr']
_C.registration = True
_C.working_dir = osp.dirname(osp.abspath(__file__))
_C.root_dir = osp.dirname(_C.working_dir)
_C.exp_name = '_'.join(_C.modules)
_C.output_dir = osp.join(_C.root_dir, 'output', _C.exp_name)
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')
_C.log_dir = osp.join(_C.output_dir, 'logs')
_C.event_dir = osp.join(_C.output_dir, 'events')

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)

# preprocess params
_C.preprocess = edict()
_C.preprocess.pc_resolutions = [512] # [32, 64, 128, 256, 512]

# Data params
_C.data = edict()
_C.data.subscenes_per_scene = 7
_C.data.filter_segment_size = 512
_C.data.min_obj_points = 50
_C.data.name = '3RScan'
_C.data.anchor_type_name = '_subscan_anchors_w_wo_overlap'

# Training params
_C.train = edict()
_C.train.batch_size = 4
_C.train.pc_res = 512
_C.train.use_augmentation = True
_C.train.rot_factor = 1.0
_C.train.augmentation_noise = 0.005

# Validation params
_C.val = edict()
_C.val.data_mode = 'orig'
_C.val.batch_size = 4
_C.val.pc_res = 512

# model param
_C.model = edict()
_C.model.rel_dim = 41
_C.model.attr_dim = 164
_C.model.alignment_thresh = 0.4

# optim
_C.optim = edict()
_C.optim.lr = 1e-3
_C.optim.lr_decay = 0.95
_C.optim.lr_decay_steps = 1
_C.optim.weight_decay = 1e-6
_C.optim.max_epoch = 50
_C.optim.grad_acc_steps = 1

# loss
_C.loss = edict()
_C.loss.alignment_loss_weight = 1.0
_C.loss.constrastive_loss_weight = 1.0
_C.loss.zoom = 0.1

# registration model params
_C.reg_model = edict()
_C.reg_model.K = 1
_C.reg_model.neighbor_limits = [38, 36, 36, 38]
_C.reg_model.num_p2p_corrs = 20000
_C.reg_model.corr_score_thresh = 0.1
_C.reg_model.rmse_thresh = 0.2 
_C.reg_model.inlier_ratio_thresh = 0.05
_C.reg_model.ransac_threshold = 0.03
_C.reg_model.ransac_min_iters = 5000
_C.reg_model.ransac_max_iters = 5000
_C.reg_model.ransac_use_sprt = False

# inference
_C.metrics = edict()
_C.metrics.all_k = [1, 2, 3, 4, 5]

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




