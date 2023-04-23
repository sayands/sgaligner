import argparse
import os 
import os.path as osp
from tqdm import tqdm
import numpy as np

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
from utils import common, define, scan3r, alignment, open3d, registration

def load_subscan_pair(src_scan_id, ref_scan_id, mode='orig', pc_res=512):
    # Centering
    src_points = scan3r.load_plydata_npy(osp.join(define.SCAN3R_SUBSCENES_DIR, 'scenes/{}/data.npy'.format(src_scan_id)), obj_ids = None)
    pcl_center = np.mean(src_points, axis=0)

    src_data = common.load_pkl_data(osp.join(define.SCAN3R_SUBSCENES_DIR, 'files/{}/data/{}.pkl'.format(mode, src_scan_id)))
    ref_data = common.load_pkl_data(osp.join(define.SCAN3R_SUBSCENES_DIR, 'files/{}/data/{}.pkl'.format(mode, ref_scan_id)))
    
    src_objects_ids = src_data['objects_id']
    ref_objects_ids = ref_data['objects_id']
    global_obj_ids = np.concatenate((src_data['objects_cat'], ref_data['objects_cat']))

    src_edges = src_data['edges']
    ref_edges = ref_data['edges']

    src_obj_points = src_data['obj_points'][pc_res] - pcl_center
    ref_obj_points = ref_data['obj_points'][pc_res] - pcl_center
    tot_obj_points = torch.cat([torch.from_numpy(src_obj_points), torch.from_numpy(ref_obj_points)]).type(torch.FloatTensor)

    edges = torch.cat([torch.from_numpy(src_edges), torch.from_numpy(ref_edges)])

    tot_bow_vec_obj_attr_feats = torch.cat([torch.from_numpy(src_data['bow_vec_object_attr_feats']), torch.from_numpy(ref_data['bow_vec_object_attr_feats'])])
    tot_bow_vec_obj_edge_feats = torch.cat([torch.from_numpy(src_data['bow_vec_object_edge_feats']), torch.from_numpy(ref_data['bow_vec_object_edge_feats'])])
    tot_rel_pose = torch.cat([torch.from_numpy(src_data['rel_trans']), torch.from_numpy(ref_data['rel_trans'])])

    src_obj_id2idx = src_data['object_id2idx']
    src_objects_idxs = np.array([src_obj_id2idx[src_object_id] for src_object_id in src_objects_ids]) 

    data_dict = {} 
    data_dict['obj_ids'] = np.concatenate([src_objects_ids, ref_objects_ids])
    data_dict['tot_obj_pts'] = tot_obj_points
    data_dict['src_objects_idxs'] = src_objects_idxs
    data_dict['src_objects_counts'] = src_objects_idxs.shape[0]

    data_dict['tot_obj_count'] = tot_obj_points.shape[0]
    data_dict['graph_per_obj_count'] = np.array([[src_obj_points.shape[0], ref_obj_points.shape[0]]])
    data_dict['graph_per_edge_count'] = np.array([[src_edges.shape[0], ref_edges.shape[0]]])
    data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_obj_attr_feats
    data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_obj_edge_feats
    data_dict['tot_rel_pose'] = tot_rel_pose
    data_dict['edges'] = edges    
    data_dict['global_obj_ids'] = global_obj_ids
    data_dict['scene_ids'] = [src_scan_id, ref_scan_id]
    data_dict['center'] = pcl_center
    data_dict['batch_size'] = 1

    return data_dict

class MosaickTester(SingleTester):
    def __init__(self, cfg, parser):
        super().__init__(cfg, parser=parser)

        self.run_reg = cfg.registration

        # Model Specific params
        self.modules = cfg.modules
        self.rel_dim = cfg.model.rel_dim
        self.attr_dim = cfg.model.attr_dim

        # Metrics Params
        self.metrics = dict()
        self.metrics['aligner_mosaicking_metrics'] = {'prec' : [], 'recall' : [], 'fscore' : [], 'acc' : [], 'comp' : []}
        self.metrics['normal_mosaicking_metrics'] = {'prec' : [], 'recall' : [], 'fscore' : [], 'acc' : [], 'comp' : []}
        
        # model 
        model = self.create_model()
        self.register_model(model)
        self.model.eval()

        # Paths 
        self.scans_dir = osp.join(cfg.data_dir)
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')

        self.subscans_dir = osp.join(self.scans_dir, 'out')
        self.subscans_files_dir = osp.join(self.subscans_dir, 'files')
        self.subscans_scenes_dir = osp.join(self.subscans_dir, 'scenes')

        # Data - load scan-subscan mapping
        self.data_mode = cfg.val.data_mode
        self.scan_subscan_map = common.load_json(osp.join(self.subscans_files_dir,  self.data_mode, cfg.data.anchor_type_name + '_{}.json'.format(cfg.split)))
        self.scan_subscan_map = {k: self.scan_subscan_map[k] for k in list(self.scan_subscan_map)[:2]}

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
            metrics_dict = self.compute_metrics(results_dict[key])
            message = common.get_log_string(result_dict=metrics_dict, name=key, timer=self.timer)
            self.logger.critical(message)
        
    def compute_metrics(self, result_dict):
        metrics_dict = {}
        for key in result_dict:
            metrics_dict[key] = round(np.array(result_dict[key]).mean(), 5)
        
        return metrics_dict
    
    def run_pairwise_alignment(self, data_dict):
        with torch.no_grad():
            data_dict = torch_util.to_cuda(data_dict)
            output_emb = self.model(data_dict)['joint'] if len(self.modules) > 1 else self.model(data_dict)[self.modules[0]]

        src_objects_count = data_dict['graph_per_obj_count'][0][0]
        ref_objects_count = data_dict['graph_per_obj_count'][0][1]
        all_obj_ids = data_dict['obj_ids']
        center = data_dict['center']

        output_emb = output_emb / output_emb.norm(dim=1)[:, None]
        dist = 1 - torch.mm(output_emb, output_emb.transpose(0,1))
        rank_list = torch.argsort(dist, dim = 1)

        node_corrs = alignment.compute_node_corrs(rank_list, src_objects_count, k=1) # TODO
        node_corrs = alignment.get_node_corrs_objects_ids(node_corrs, all_obj_ids, 0)
        data_dict['node_corrs'] = node_corrs
        alignment_score = alignment.compute_alignment_score(rank_list, src_objects_count, ref_objects_count)

        return alignment_score, data_dict

    def eval(self):
        total_iterations = len(self.scan_subscan_map)
        pbar = tqdm(enumerate(self.scan_subscan_map), total=total_iterations)     

        for iteration, scan_id in pbar:
            subscan_ids = self.scan_subscan_map[scan_id]

            if len(subscan_ids) == 0: continue
            
            origin_subscan_id = subscan_ids[0]

            origin_subscan_path = osp.join(self.subscans_scenes_dir , origin_subscan_id, 'data.npy')
            origin_subscan_points = scan3r.load_plydata_npy(origin_subscan_path)

            reconstructed_scan_pcd_aligner = open3d.make_open3d_point_cloud(origin_subscan_points)
            reconstructed_scan_pcd_registration = open3d.make_open3d_point_cloud(origin_subscan_points)
            gt_scan_pcd = open3d.make_open3d_point_cloud(origin_subscan_points)

            for src_subscan_id in subscan_ids[1:]:
                data_dict = load_subscan_pair(src_subscan_id, origin_subscan_id)
                alignment_score, data_dict = self.run_pairwise_alignment(data_dict)

                src_points, src_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, src_subscan_id, 'data.npy'), obj_ids=None, return_ply_data=True)
                ref_points, ref_plydata = scan3r.load_plydata_npy(osp.join(self.subscans_scenes_dir, origin_subscan_id, 'data.npy'), obj_ids=None, return_ply_data=True)

                gt_scan_pcd += open3d.make_open3d_point_cloud(src_points)

                reg_data_dict = dict()
                reg_data_dict['node_corrs'] = data_dict['node_corrs']
                reg_data_dict['src_points'] = src_points - data_dict['center']
                reg_data_dict['ref_points'] = ref_points - data_dict['center']
                reg_data_dict['src_plydata'] = src_plydata
                reg_data_dict['ref_plydata'] = ref_plydata 
                reg_data_dict['gt_transform'] = np.eye(4)

                est_transform_aligner = self.registration_evaluator.run_aligner_registration(reg_data_dict, evaluate_registration=False)
                registration_res = self.registration_evaluator.run_normal_registration(reg_data_dict, evaluate_registration=False)

                if registration_res is None: continue

                est_transform_registration_method = registration_res[0]

                if est_transform_aligner is None or est_transform_registration_method is None: continue

                src_pcd_transformed_aligner = open3d.make_open3d_point_cloud(registration.apply_transform(src_points, est_transform_aligner))
                src_pcd_transformed_registration_method = open3d.make_open3d_point_cloud(registration.apply_transform(src_points, est_transform_registration_method))
                reconstructed_scan_pcd_aligner += src_pcd_transformed_aligner
                reconstructed_scan_pcd_registration += src_pcd_transformed_registration_method
            
            reconstructed_scan_points_aligner = np.asarray(reconstructed_scan_pcd_aligner.points)
            reconstructed_scan_points_registration_method = np.asarray(reconstructed_scan_pcd_registration.points)
            gt_scene_points = np.asarray(gt_scan_pcd.points)

            self.metrics['aligner_mosaicking_metrics'] = common.update_dict(self.metrics['aligner_mosaicking_metrics'], 
                                                                            registration.compute_mosaicking_error(reconstructed_scan_points_aligner, gt_scene_points))
            self.metrics['normal_mosaicking_metrics'] = common.update_dict(self.metrics['normal_mosaicking_metrics'], 
                                                                            registration.compute_mosaicking_error(reconstructed_scan_points_registration_method, gt_scene_points))
        
        results_dict = dict()
        results_dict['aligner_mosaicking_metrics'] = self.metrics['aligner_mosaicking_metrics']
        results_dict['normal_mosaicking_metrics'] = self.metrics['normal_mosaicking_metrics']
        self.print_metrics(results_dict)

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
    tester = MosaickTester(cfg, parser)
    tester.eval()

if __name__ == '__main__':
    main()