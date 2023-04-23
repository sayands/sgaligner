import argparse
import os 
import os.path as osp
import time
import numpy as np 
from sklearn.metrics import confusion_matrix

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append('.')

from engine.single_tester import SingleTester
from engine.registration_evaluator import RegistrationEvaluator
from utils import torch_util
from aligner.sg_aligner import *
from datasets.loaders import get_val_dataloader
from configs import config, update_config
from utils import common, scan3r, alignment

class AlignerOverlapper(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)

        self.run_reg = cfg.registration

        # Model Specific params
        self.modules = cfg.modules
        self.rel_dim = cfg.model.rel_dim
        self.attr_dim = cfg.model.attr_dim

        # Metrics params
        self.alignment_thresh = cfg.model.alignment_thresh
        self.corr_score_thresh = cfg.reg_model.corr_score_thresh

        self.aligner_overlapper_data = {'true' : [], 'pred' : []}
        self.registration_overlapper_data = {'true' : [], 'pred' : []}

        # dataloader
        start_time = time.time()
        dataset, data_loader = get_val_dataloader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)

        self.register_loader(data_loader)
        self.register_dataset(dataset)

        # model 
        model = self.create_model()
        self.register_model(model)
        self.model.eval()

        # Registration
        self.reg_k = cfg.reg_model.K
        reg_snapshot = self.args.reg_snapshot
        self.registration_evaluator = RegistrationEvaluator(self.device, cfg, reg_snapshot, self.logger)

    def create_model(self):
        model = MultiModalEncoder(modules = self.modules, rel_dim = self.rel_dim, attr_dim=self.attr_dim).to(self.device)
        message = 'Model created'
        self.logger.info(message)
        return model
    
    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict
    
    def print_metrics(self, results_dict):
        for key in results_dict.keys():
            metrics_dict = self.compute_precision_recall(results_dict[key])
            message = common.get_log_string(result_dict=metrics_dict, name=key, timer=self.timer)
            self.logger.critical(message)

    def compute_precision_recall(self, result_dict):
        tn, fp, fn, tp = confusion_matrix(result_dict['true'], result_dict['pred'], labels=[0, 1]).ravel()
        precision = round(tp / (tp + fp), 4)
        recall = round(tp / (tp + fn), 4)
        f1_score = round(2 * (precision * recall)/ (precision + recall), 4)
        metrics_dict = {'precision' : precision, 'recall' : recall, 'f1_score' : f1_score}

        return metrics_dict
        
    def eval_step(self, iteration, data_dict, output_dict):
        obj_cnt_start_idx = 0
        data_dict = torch_util.release_cuda(data_dict)
        embedding = output_dict['joint'] if len(self.modules) > 1 else output_dict[self.modules[0]]

        for batch_idx in range(self.test_loader.batch_size):
            src_objects_count = data_dict['graph_per_obj_count'][batch_idx][0]
            ref_objects_count = data_dict['graph_per_obj_count'][batch_idx][1]
            overlap = data_dict['overlap'][batch_idx]
            obj_cnt_end_idx = obj_cnt_start_idx + data_dict['tot_obj_count'][batch_idx]
            pcl_center = data_dict['pcl_center'][batch_idx]

            embedding_batch_idx = embedding[obj_cnt_start_idx : obj_cnt_end_idx]
            embedding_batch_idx /= embedding_batch_idx.norm(dim=1)[:, None]
            dist = 1 - torch.mm(embedding_batch_idx, embedding_batch_idx.transpose(0, 1))
            rank_list = torch.argsort(dist, dim=1)
            
            # Load subscene points
            src_scan_id = data_dict['scene_ids'][batch_idx][0]
            ref_scan_id = data_dict['scene_ids'][batch_idx][1]

            src_points = scan3r.load_plydata_npy(osp.join(self.test_dataset.subscans_scenes_dir, src_scan_id, 'data.npy'), obj_ids=None, return_ply_data=False)
            ref_points = scan3r.load_plydata_npy(osp.join(self.test_dataset.subscans_scenes_dir, ref_scan_id, 'data.npy'), obj_ids=None, return_ply_data=False)

            reg_data_dict = dict()
            reg_data_dict['src_points'] = src_points - pcl_center
            reg_data_dict['ref_points'] = ref_points - pcl_center
            reg_data_dict['gt_transform'] = np.eye(4)
            corr_score = self.registration_evaluator.run_normal_registration(reg_data_dict, evaluate_registration=False)

            if corr_score is not None:
                alignment_score = alignment.compute_alignment_score(rank_list, src_objects_count, ref_objects_count)
                
                self.registration_overlapper_data['pred'].append(1.0 if corr_score > self.corr_score_thresh else 0.0)
                self.registration_overlapper_data['true'].append(1.0 if overlap > 0.0 else 0.0)
                
                self.aligner_overlapper_data['pred'].append(1.0 if alignment_score > self.alignment_thresh else 0.0)
                self.aligner_overlapper_data['true'].append(1.0 if overlap > 0.0 else 0.0)
        
            obj_cnt_start_idx = obj_cnt_end_idx
        
        return { 'aligner_overlapper_data' : self.aligner_overlapper_data, 'registration_overlapper_data' : self.registration_overlapper_data}

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--test_epoch', type=int, default=None, help='test epoch')
    parser.add_argument('--test_iter', type=int, default=None, help='test iteration')
    parser.add_argument('--reg_snapshot', default=None, help='load from snapshot')

    args = parser.parse_args()
    return parser, args
    
def main():
    parser, args = parse_args()
    cfg = update_config(config, args.config)
    tester = AlignerOverlapper(cfg, parser)
    tester.run()

if __name__ == '__main__':
    main()