import argparse
import os 
import os.path as osp
import time
import numpy as np 

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append('.')

from engine.single_tester import SingleTester
from utils import torch_util
from aligner.eva import *
from datasets.loaders import get_val_dataloader
from configs import config, update_config
from utils import alignment, common, point_cloud

class EVATester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)
        # Model Specific params
        self.modules = cfg.modules
        self.rel_dim = cfg.model.rel_dim
        self.attr_dim = cfg.model.attr_dim

        # Metrics params
        self.all_k = cfg.metrics.all_k
        self.alignment_metrics_meter = {'mrr' : []}
        for k in self.all_k:
            self.alignment_metrics_meter[k] = {'correct' : 0, 'total' : 0}
        
        self.normal_registration_metrics_meter = {'CD' : [], 'IR' : [], 'RRE' : [], 'RTE' : [], 'recall' : [], 'FMR' : []}
        self.aligner_registration_metrics_meter = {'CD' : [], 'IR' : [], 'RRE' : [], 'RTE' : [], 'recall' : [], 'FMR' : []}

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

    def create_model(self):
        model = EVA(modules = self.modules, rel_dim = self.rel_dim, attr_dim=self.attr_dim).to(self.device)
        message = 'Model created'
        self.logger.info(message)
        return model
    
    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict
    
    def print_metrics(self, results_dict):
        for key in results_dict.keys():
            metrics_dict = self.compute_metrics(results_dict[key])
            message = common.get_log_string(result_dict=metrics_dict, name=key, timer=self.timer)
            self.logger.critical(message)

    def compute_metrics(self, result_dict):
        metrics_dict = {}
        for key in result_dict:
            if type(key) == int:
                metrics_dict['hits@_{}'.format(key)] = round(result_dict[key]['correct'] / result_dict[key]['total'], 5)
            else:
                metrics_dict[key] = round(np.array(result_dict[key]).mean(), 5)
        
        return metrics_dict
        
    def eval_step(self, iteration, data_dict, output_dict):
        data_dict = torch_util.release_cuda(data_dict)
        embedding = output_dict['joint'] if len(self.modules) > 1 else embedding[self.modules[0]]

        e1i_start_idx = 0
        e2i_start_idx = 0
        obj_cnt_start_idx = 0
        curr_total_objects_count = 0    
        
        for batch_idx in range(self.test_loader.batch_size):
            src_objects_count = data_dict['graph_per_obj_count'][batch_idx][0]
            ref_objects_count = data_dict['graph_per_obj_count'][batch_idx][1]
            pcl_center = data_dict['pcl_center'][batch_idx]
            
            all_objects_ids = data_dict['obj_ids']
            e1i_end_idx = e1i_start_idx + data_dict['e1i_count'][batch_idx]
            e2i_end_idx = e2i_start_idx + data_dict['e2i_count'][batch_idx]
            obj_cnt_end_idx = obj_cnt_start_idx + data_dict['tot_obj_count'][batch_idx]

            e1i_idxs = data_dict['e1i'][e1i_start_idx : e1i_end_idx]
            e2i_idxs = data_dict['e2i'][e2i_start_idx : e2i_end_idx]
            e1i_idxs -= curr_total_objects_count
            e2i_idxs -= curr_total_objects_count

            if e1i_idxs.shape[0] != 0 and e2i_idxs.shape[0] != 0:
                assert e1i_idxs.shape == e2i_idxs.shape
                
                emb = embedding[obj_cnt_start_idx : obj_cnt_end_idx]
                emb = emb / emb.norm(dim=1)[:, None]
                dist = 1 - torch.mm(emb, emb.transpose(0,1))
                rank_list = torch.argsort(dist, dim = 1)
                assert np.max(e1i_idxs) <= rank_list.shape[0]

                # Compute Mean Reciprocal Rank
                self.alignment_metrics_meter['mrr'] = alignment.compute_mean_reciprocal_rank(rank_list, e1i_idxs, e2i_idxs, self.alignment_metrics_meter['mrr'] )

                # Compute Hits@k = {1, 2, 3, 4, 5}
                for k in self.all_k:
                    correct, total = alignment.compute_hits_k(rank_list, e1i_idxs, e2i_idxs, k)
                    self.alignment_metrics_meter[k]['correct'] += correct
                    self.alignment_metrics_meter[k]['total'] += total

            obj_cnt_start_idx = obj_cnt_end_idx
            curr_total_objects_count += data_dict['tot_obj_count'][batch_idx]
            e1i_start_idx, e2i_start_idx = e1i_end_idx, e2i_end_idx
            
        return { 'alignment_metrics' : self.alignment_metrics_meter, 'normal_registration_metrics' : self.normal_registration_metrics_meter,
                'aligner_registration_metrics' : self.aligner_registration_metrics_meter }

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
    cfg = update_config(config, args.config, ensure_dir=True)

    tester = EVATester(cfg, parser)
    tester.run()

if __name__ == '__main__':
    main()