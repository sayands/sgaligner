import argparse
import os 
import os.path as osp
import time
import numpy as np 

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from engine.single_tester import SingleTester
from engine.registration_evaluator import RegistrationEvaluator
from utils import torch_util, scan3r, registration
from aligner.sg_aligner import *
from datasets.loaders import get_val_dataloader
from configs.config_scan3r_gt import make_cfg
from utils import alignment, common, point_cloud

def make_parser():
    parser = argparse.ArgumentParser()
    return parser

class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg, parser=make_parser())

        self.run_reg = cfg.registration

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

        # Registration
        self.reg_k = cfg.reg_model.K
        reg_snapshot = self.args.reg_snapshot
        self.registration_evaluator = RegistrationEvaluator(self.device, cfg, reg_snapshot, self.logger, visualise_registration=True)
        self.visualise_registration = True

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
            if not self.run_reg and  'registration' in key: continue
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
        e1i_start_idx = 0
        e2i_start_idx = 0
        obj_cnt_start_idx = 0
        curr_total_objects_count = 0

        data_dict = torch_util.release_cuda(data_dict)
        embedding = output_dict['joint'] if len(self.modules) > 1 else output_dict[self.modules[0]]

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

            if e1i_idxs.shape[0] == 0 or e2i_idxs.shape[0] == 0: continue

            assert e1i_idxs.shape[0] == e2i_idxs.shape[0]

            embedding_batch_idx = embedding[obj_cnt_start_idx : obj_cnt_end_idx]
            embedding_batch_idx /= embedding_batch_idx.norm(dim=1)[:, None]
            dist = 1 - torch.mm(embedding_batch_idx, embedding_batch_idx.transpose(0, 1))
            rank_list = torch.argsort(dist, dim=1)
            assert np.max(e1i_idxs) <= rank_list.shape[0]

            # Compute Mean Reciprocal Rank
            self.alignment_metrics_meter['mrr'] = alignment.compute_mean_reciprocal_rank(rank_list, e1i_idxs, e2i_idxs, self.alignment_metrics_meter['mrr'] )

            # Compute Hits@k = {1, 2, 3, 4, 5}
            for k in self.all_k:
                correct, total = alignment.compute_hits_k(rank_list, e1i_idxs, e2i_idxs, k)
                self.alignment_metrics_meter[k]['correct'] += correct
                self.alignment_metrics_meter[k]['total'] += total
            
            if self.run_reg:
                node_corrs = alignment.compute_node_corrs(rank_list, e1i_idxs, src_objects_count, k=self.reg_k)
                node_corrs = alignment.get_node_corrs_objects_ids(node_corrs, all_objects_ids, curr_total_objects_count)

                # Load subscene points
                src_scan_id = data_dict['scene_ids'][batch_idx][0]
                ref_scan_id = data_dict['scene_ids'][batch_idx][1]
                scan_id = src_scan_id[:src_scan_id.index('_')]

                src_points, src_plydata = scan3r.load_plydata_npy(osp.join(self.test_dataset.subscans_scenes_dir, src_scan_id, 'data.npy'), obj_ids=None, return_ply_data=True)
                ref_points, ref_plydata = scan3r.load_plydata_npy(osp.join(self.test_dataset.subscans_scenes_dir, ref_scan_id, 'data.npy'), obj_ids=None, return_ply_data=True)
                raw_points = scan3r.load_plydata_npy(osp.join(self.test_dataset.scans_scenes_dir, scan_id, 'data.npy'))
                
                reg_data_dict = dict()
                reg_data_dict['node_corrs'] = node_corrs
                reg_data_dict['src_points'] = src_points - pcl_center
                reg_data_dict['ref_points'] = ref_points - pcl_center
                reg_data_dict['src_plydata'] = src_plydata
                reg_data_dict['ref_plydata'] = ref_plydata 
                reg_data_dict['raw_points'] = raw_points - pcl_center
                reg_data_dict['gt_transform'] = np.eye(4)

                _, gt_src_corr_idxs = point_cloud.compute_pcl_overlap(reg_data_dict['src_points'], reg_data_dict['ref_points'] )
                _, gt_ref_corr_idxs = point_cloud.compute_pcl_overlap(reg_data_dict['ref_points'] , reg_data_dict['src_points'])

                reg_data_dict['gt_src_corr_points'] = reg_data_dict['src_points'][gt_src_corr_idxs]
                reg_data_dict['gt_ref_corr_points'] = reg_data_dict['ref_points'] [gt_ref_corr_idxs]

                all_reg_results_dict = self.registration_evaluator.run_registration(reg_data_dict)
                normal_reg_results_dict = all_reg_results_dict[0]
                aligner_reg_results_dict = all_reg_results_dict[1]

                if normal_reg_results_dict is not None and aligner_reg_results_dict is not None:
                    self.aligner_registration_metrics_meter = common.update_dict(self.aligner_registration_metrics_meter, aligner_reg_results_dict)
                    self.normal_registration_metrics_meter = common.update_dict(self.normal_registration_metrics_meter, normal_reg_results_dict)

            obj_cnt_start_idx = obj_cnt_end_idx
            curr_total_objects_count += data_dict['tot_obj_count'][batch_idx]
            e1i_start_idx, e2i_start_idx = e1i_end_idx, e2i_end_idx
        
        return { 'alignment_metrics' : self.alignment_metrics_meter, 'normal_registration_metrics' : self.normal_registration_metrics_meter,
                 'aligner_registration_metrics' : self.aligner_registration_metrics_meter }



def main():
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()

if __name__ == '__main__':
    main()