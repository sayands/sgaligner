import argparse
import os 
import os.path as osp
import time
import numpy as np 

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from engine.single_tester import SingleTester
from utils import torch_util
from aligner.sg_aligner import *
from datasets.loaders import get_val_dataloader
from configs.config_scan3r_gt import make_cfg
from utils import alignment_metrics

def make_parser():
    parser = argparse.ArgumentParser()
    return parser

class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        # Model Specific params
        self.modules = cfg.modules
        self.rel_dim = cfg.model.rel_dim
        self.attr_dim = cfg.model.attr_dim

        # Metrics params
        self.all_k = cfg.metrics.all_k

        # dataloader
        start_time = time.time()
        data_loader = get_val_dataloader(cfg)
        loading_time = time.time() - start_time
        message = f'Data loader created: {loading_time:.3f}s collapsed.'
        self.logger.info(message)

        self.register_loader(data_loader)

        # model 
        model = self.create_model()
        self.register_model(model)
        self.model.eval()

    def create_model(self):
        model = MultiModalEncoder(modules = self.modules, rel_dim = self.rel_dim, attr_dim=self.attr_dim).to(self.device)
        message = 'Model created'
        self.logger.info(message)
        return model
    
    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict
    
    def compute_metrics(self, result_dict):
        metrics_dict = {}
        for key in result_dict:
            if key == 'mrr':
                metrics_dict[key] = np.array(result_dict[key]).mean()
            else:
                metrics_dict['hits@_{}'.format(key)] = round(result_dict[key]['correct'] / result_dict[key]['total'], 4)
        
        return metrics_dict
        
    def eval_step(self, iteration, data_dict, output_dict):
        e1i_start_idx = 0
        e2i_start_idx = 0
        obj_cnt_start_idx = 0
        curr_total_objects_count = 0

        data_dict = torch_util.release_cuda(data_dict)
        embedding = output_dict['joint'] if len(self.modules) > 1 else output_dict[self.modules[0]]

        # Add metrics
        alignment_metrics_meter = {'mrr' : []}
        for k in self.all_k:
            alignment_metrics_meter[k] = {'correct' : 0, 'total' : 0}
        
        for batch_idx in range(self.test_loader.batch_size):
            src_obj_count = data_dict['graph_per_obj_count'][batch_idx][0]
            ref_obj_count = data_dict['graph_per_obj_count'][batch_idx][1]

            all_obj_ids = data_dict['obj_ids']
            e1i_end_idx = e1i_start_idx + data_dict['e1i_count'][batch_idx]
            e2i_end_idx = e2i_start_idx + data_dict['e2i_count'][batch_idx]
            obj_cnt_end_idx = obj_cnt_start_idx + data_dict['tot_obj_count'][batch_idx]

            e1i_idxs = data_dict['e1i'][e1i_start_idx : e1i_end_idx]
            e2i_idxs = data_dict['e2i'][e2i_start_idx : e2i_end_idx]
            
            e1i_idxs -= curr_total_objects_count
            e2i_idxs -= curr_total_objects_count

            if e1i_idxs.shape[0] == 0 or e2i_idxs.shape[0] == 0: continue

            assert e1i_idxs.shape[0] == e2i_idxs.shape[0]

            embedding_batch_idx = embedding[obj_cnt_start_idx : obj_cnt_end_idx]
            embedding_batch_idx /= embedding_batch_idx.norm(dim=1)[:, None]
            dist = 1 - torch.mm(embedding_batch_idx, embedding_batch_idx.transpose(0, 1))
            rank_list = torch.argsort(dist, dim=1)
            assert np.max(e1i_idxs) <= rank_list.shape[0]

            # Compute Mean Reciprocal Rank
            alignment_metrics_meter['mrr'] = alignment_metrics.compute_mean_reciprocal_rank(rank_list, e1i_idxs, e2i_idxs, alignment_metrics_meter['mrr'] )

            # Compute Hits@k = {1, 2, 3, 4, 5}
            for k in self.all_k:
                correct, total = alignment_metrics.compute_hits_k(rank_list, e1i_idxs, e2i_idxs, k)
                alignment_metrics_meter[k]['correct'] += correct
                alignment_metrics_meter[k]['total'] += total

            obj_cnt_start_idx = obj_cnt_end_idx
            curr_total_objects_count += data_dict['tot_obj_count'][batch_idx]
            e1i_start_idx, e2i_start_idx = e1i_end_idx, e2i_end_idx
        return alignment_metrics_meter

def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()

if __name__ == '__main__':
    main()