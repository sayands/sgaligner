import sys
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn

from utils import torch_util, registration

def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--reg_snapshot', default=None, help='load from snapshot')
    return parser

class RegistrationEvaluator(nn.Module):
    def __init__(self, device,  cfg, parser=None):
        super(RegistrationEvaluator, self).__init__()
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()

        self.device = device
        
        # Load Registration Model
        self.cfg = make_cfg()
        self.model = create_model().to(self.device) TODO
        self.model.eval()
        state_dict_path = self.args.reg_snapshot
        state_dict = torch.load(state_dict_path)
        self.model.load_state_dict(state_dict['model'])

        # Params
        self.num_p2p_corrs
        self.neighbor_limits
        self.ransac_threshold = 
        self.ransac_min_iters = 5000
        self.ransac_max_iters = 5000
        self.ransac_use_sprt = False

    @torch.no_grad
    def evaluate_registration(self, src_points, ref_points, raw_points, pose, gt_transform, corrs_ransac, gt_src_corr_points, gt_ref_corr_points):
        chamfer_distance = registration.compute_modified_chamfer_distance(src_points, ref_points, raw_points, pose, gt_transform)
        inlier_ratio = registration.compute_inlier_ratio(corrs_ransac[:, 3:], corrs_ransac[:, :3], gt_transform)
        rre, rte = registration.compute_registration_error(gt_transform, pose, inverse_trans=True) 
        registration_rmse = registration.compute_registration_rmse(gt_src_corr_points , gt_ref_corr_points, pose)

        return chamfer_distance, inlier_ratio, rre, rte, registration_rmse

    @torch.no_grad()
    def run(self, data_dict):
        pcl_center = data_dict['pcl_center']
        node_corrs = data_dict['node_corrs']
        src_points = data_dict['src_points'] - pcl_center
        ref_points = data_dict['ref_points'] - pcl_center
        raw_points = data_dict['raw_points'] - pcl_center
        src_plydata = data_dict['src_plydata'] 
        ref_plydata = data_dict['ref_plydata']
        gt_transform = data_dict['gt_transform']
        
        
        point_corrs = []
        for node_corr in node_corrs:
            node_points_src = src_points[np.where(src_plydata['objectId']  == node_corr[0])[0]]
            node_points_ref = ref_points[np.where(ref_plydata['objectId']  == node_corr[1])[0]]

            src_feats = np.ones_like(node_points_src[:, :1])
            ref_feats = np.ones_like(node_points_ref[:, :1])

            data_dict = {
                "ref_points": node_points_ref.astype(np.float32),
                "src_points": node_points_src.astype(np.float32),
                "ref_feats": ref_feats.astype(np.float32),
                "src_feats": src_feats.astype(np.float32),
                "transform" : gt_transform.astype(np.float32)
            }

            with torch.no_grad():
                data_dict = registration_collate_fn_stack_mode([data_dict], 
                                self.cfg.backbone.num_stages, self.cfg.backbone.init_voxel_size, 
                                self.cfg.backbone.init_radius, self.neighbor_limits)
                
                # output dict
                data_dict = torch_util.to_cuda(data_dict)
                try:
                    output_dict = self.registration_model(data_dict)
                except:
                    continue
            
            output_dict = torch_util.release_cuda(output_dict)
            ref_corr_points = output_dict['ref_corr_points']
            src_corr_points = output_dict['src_corr_points']
            corr_scores = output_dict['corr_scores']

            if corr_scores.shape[0] > self.num_p2p_corrs // len(node_corrs):
                sel_indices = np.argsort(-corr_scores)[: self.num_p2p_corrs // len(node_corrs)]
                ref_corr_points = ref_corr_points[sel_indices]
                src_corr_points = src_corr_points[sel_indices]

            point_corrs['src'].append(src_corr_points)
            point_corrs['ref'].append(ref_corr_points)
            point_corrs['scores'].append(corr_scores)

        point_corrs['src'] = np.concatenate(point_corrs['src'])
        point_corrs['ref'] = np.concatenate(point_corrs['ref'])

        corrs_ransac = np.concatenate([point_corrs['src'], point_corrs['ref']], axis=1)
        
        if corrs_ransac.shape[0] > self.num_p2p_corrs:
            corr_sel_indices = np.random.choice(corrs_ransac.shape[0], self.num_p2p_corrs)
            corrs_ransac = corrs_ransac[corr_sel_indices]
        
        est_transform, _ = pygcransac.findRigidTransform(np.ascontiguousarray(corrs_ransac), probabilities = [], threshold = self.ransac_threshold, neighborhood_size = 4, 
                                                         sampler = 1, min_iters = self.ransac_min_iters, max_iters = self.ransac_max_iters, spatial_coherence_weight = 0.0, 
                                                         use_space_partitioning = not self.ransac_use_sprt, neighborhood = 0, conf = 0.999, use_sprt = self.ransac_use_sprt)
        if est_transform is None: return None

        chamfer_distance, inlier_ratio, rre, rte, registration_rmse = self.evaluate_registration(src_points, ref_points, raw_points, est_transform, gt_transform, 
                                                                                                corrs_ransac, gt_src_corr_points, gt_ref_corr_points)
        return {
            'CD' : chamfer_distance,
            'IR' : inlier_ratio,
            'RRE' : rre,
            'RTE' : rte,
            'Reg_RMSE' : registration_rmse
        }
