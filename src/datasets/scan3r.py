import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

import sys
sys.path.append('..')

from utils import common, scan3r

class Scan3RDataset(data.Dataset):
    def __init__(self, cfg, split):
        self.split = split
        self.use_predicted = cfg.use_predicted
        self.pc_resolution = cfg.val.pc_res if split == 'val' else cfg.train.pc_res
        self.anchor_type_name = cfg.preprocess.anchor_type_name
        self.model_name = cfg.model_name
        self.scan_type = cfg.scan_type
        self.data_root_dir = cfg.data.root_dir
        
        scan_dirname = '' if self.scan_type == 'scan' else 'out'
        scan_dirname = osp.join(scan_dirname, 'predicted') if self.use_predicted else scan_dirname

        self.scans_dir = osp.join(cfg.data.root_dir, scan_dirname)
        self.scans_scenes_dir = osp.join(self.scans_dir, 'scenes')
        self.scans_files_dir = osp.join(self.scans_dir, 'files')
        
        self.mode = 'orig' if self.split == 'train' else cfg.val.data_mode

        self.anchor_data_filename = osp.join(self.scans_files_dir, '{}/anchors{}_{}.json'.format(self.mode, self.anchor_type_name, split))
        print('[INFO] Reading from {} with point cloud resolution - {}'.format(self.anchor_data_filename, self.pc_resolution))
        self.anchor_data = common.load_json(self.anchor_data_filename)[:]
        
        if split == 'val' and cfg.val.overlap_low != cfg.val.overlap_high:
            final_anchor_data = []
            for anchor_data_idx in self.anchor_data:
                if anchor_data_idx['overlap'] >= cfg.val.overlap_low and anchor_data_idx['overlap'] < cfg.val.overlap_high:
                    final_anchor_data.append(anchor_data_idx)
            
            self.anchor_data = final_anchor_data
        
        self.is_training = self.split == 'train'
        self.do_augmentation = False if self.split == 'val' else cfg.train.use_augmentation

        self.rot_factor = cfg.train.rot_factor
        self.augment_noise = cfg.train.augmentation_noise

        # Jitter
        self.scale = 0.01
        self.clip = 0.05

        # Random Rigid Transformation
        self._rot_mag = 45.0
        self._trans_mag = 0.5
        
    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        graph_data = self.anchor_data[idx]
        src_scan_id = graph_data['src']
        ref_scan_id = graph_data['ref']
        overlap = graph_data['overlap'] if 'overlap' in graph_data else -1.0
        
        # Centering
        src_points = scan3r.load_plydata_npy(osp.join(self.scans_scenes_dir, '{}/data.npy'.format(src_scan_id)), obj_ids = None)
        ref_points = scan3r.load_plydata_npy(osp.join(self.scans_scenes_dir, '{}/data.npy'.format(ref_scan_id)), obj_ids = None)

        if self.split == 'train':
            if np.random.rand(1)[0] > 0.5:
                pcl_center = np.mean(src_points, axis=0)
            else:
                pcl_center = np.mean(ref_points, axis=0)
        else:
            pcl_center = np.mean(src_points, axis=0)

        src_data_dict = common.load_pkl_data(osp.join(self.scans_files_dir, '{}/data/{}.pkl'.format(self.mode, src_scan_id)))
        ref_data_dict = common.load_pkl_data(osp.join(self.scans_files_dir, '{}/data/{}.pkl'.format(self.mode, ref_scan_id)))
        
        src_object_ids = src_data_dict['objects_id']
        ref_object_ids = ref_data_dict['objects_id']
        anchor_obj_ids = graph_data['anchorIds'] if 'anchorIds' in graph_data else src_object_ids
        global_object_ids = np.concatenate((src_data_dict['objects_cat'], ref_data_dict['objects_cat']))
        
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id != 0]
        anchor_obj_ids = [anchor_obj_id for anchor_obj_id in anchor_obj_ids if anchor_obj_id in src_object_ids and anchor_obj_id in ref_object_ids]
        
        if self.split == 'train':
            anchor_cnt = 2 if int(0.3 * len(anchor_obj_ids)) < 1 else int(0.3 * len(anchor_obj_ids))
            anchor_obj_ids = anchor_obj_ids[:anchor_cnt]

        src_edges = src_data_dict['edges']
        ref_edges = ref_data_dict['edges']

        src_object_points = src_data_dict['obj_points'][self.pc_resolution] - pcl_center
        ref_object_points = ref_data_dict['obj_points'][self.pc_resolution] - pcl_center

        edges = torch.cat([torch.from_numpy(src_edges), torch.from_numpy(ref_edges)])

        src_object_id2idx = src_data_dict['object_id2idx']
        e1i_idxs = np.array([src_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) # e1i
        e1j_idxs = np.array([src_object_id2idx[object_id] for object_id in src_data_dict['objects_id'] if object_id not in anchor_obj_ids]) # e1j
        
        ref_object_id2idx = ref_data_dict['object_id2idx']
        e2i_idxs = np.array([ref_object_id2idx[anchor_obj_id] for anchor_obj_id in anchor_obj_ids]) + src_object_points.shape[0] # e2i
        e2j_idxs = np.array([ref_object_id2idx[object_id] for object_id in ref_data_dict['objects_id'] if object_id not in anchor_obj_ids]) + src_object_points.shape[0] # e2j

        tot_object_points = torch.cat([torch.from_numpy(src_object_points), torch.from_numpy(ref_object_points)]).type(torch.FloatTensor)
        tot_bow_vec_obj_edge_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_edge_feats']), torch.from_numpy(ref_data_dict['bow_vec_object_edge_feats'])])
        if not self.use_predicted:
            tot_bow_vec_obj_attr_feats = torch.cat([torch.from_numpy(src_data_dict['bow_vec_object_attr_feats']), torch.from_numpy(ref_data_dict['bow_vec_object_attr_feats'])])
        
        else:
            tot_bow_vec_obj_attr_feats = torch.zeros(tot_object_points.shape[0], 41)
        
        tot_rel_pose = torch.cat([torch.from_numpy(src_data_dict['rel_trans']), torch.from_numpy(ref_data_dict['rel_trans'])])

        data_dict = {} 
        data_dict['obj_ids'] = np.concatenate([src_object_ids, ref_object_ids])
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['graph_per_obj_count'] = np.array([src_object_points.shape[0], ref_object_points.shape[0]])
        data_dict['graph_per_edge_count'] = np.array([src_edges.shape[0], ref_edges.shape[0]])
        
        data_dict['e1i'] = e1i_idxs
        data_dict['e1i_count'] = e1i_idxs.shape[0]
        data_dict['e2i'] = e2i_idxs
        data_dict['e2i_count'] = e2i_idxs.shape[0]
        data_dict['e1j'] = e1j_idxs
        data_dict['e1j_count'] = e1j_idxs.shape[0]
        data_dict['e2j'] = e2j_idxs
        data_dict['e2j_count'] = e2j_idxs.shape[0]
        
        data_dict['tot_obj_count'] = tot_object_points.shape[0]
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_obj_attr_feats
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_obj_edge_feats
        data_dict['tot_rel_pose'] = tot_rel_pose
        data_dict['edges'] = edges    

        data_dict['global_obj_ids'] = global_object_ids
        data_dict['scene_ids'] = [src_scan_id, ref_scan_id]        
        data_dict['pcl_center'] = pcl_center
        data_dict['overlap'] = overlap
        
        return data_dict

    def _collate_entity_idxs(self, batch):
        e1i = np.concatenate([data['e1i'] for data in batch])
        e2i = np.concatenate([data['e2i'] for data in batch])
        e1j = np.concatenate([data['e1j'] for data in batch])
        e2j = np.concatenate([data['e2j'] for data in batch])

        e1i_start_idx = 0 
        e2i_start_idx = 0 
        e1j_start_idx = 0 
        e2j_start_idx = 0 
        prev_obj_cnt = 0
        
        for idx in range(len(batch)):
            e1i_end_idx = e1i_start_idx + batch[idx]['e1i_count']
            e2i_end_idx = e2i_start_idx + batch[idx]['e2i_count']
            e1j_end_idx = e1j_start_idx + batch[idx]['e1j_count']
            e2j_end_idx = e2j_start_idx + batch[idx]['e2j_count']

            e1i[e1i_start_idx : e1i_end_idx] += prev_obj_cnt
            e2i[e2i_start_idx : e2i_end_idx] += prev_obj_cnt
            e1j[e1j_start_idx : e1j_end_idx] += prev_obj_cnt
            e2j[e2j_start_idx : e2j_end_idx] += prev_obj_cnt
            
            e1i_start_idx, e2i_start_idx, e1j_start_idx, e2j_start_idx = e1i_end_idx, e2i_end_idx, e1j_end_idx, e2j_end_idx
            prev_obj_cnt += batch[idx]['tot_obj_count']
        
        e1i = e1i.astype(np.int32)
        e2i = e2i.astype(np.int32)
        e1j = e1j.astype(np.int32)
        e2j = e2j.astype(np.int32)

        return e1i, e2i, e1j, e2j

    def _collate_feats(self, batch, key):
        feats = torch.cat([data[key] for data in batch])
        return feats
    
    def collate_fn(self, batch):
        tot_object_points = self._collate_feats(batch, 'tot_obj_pts')
        tot_bow_vec_object_attr_feats = self._collate_feats(batch, 'tot_bow_vec_object_attr_feats')
        tot_bow_vec_object_edge_feats = self._collate_feats(batch, 'tot_bow_vec_object_edge_feats')    
        tot_rel_pose = self._collate_feats(batch, 'tot_rel_pose')
        
        data_dict = {}
        data_dict['tot_obj_pts'] = tot_object_points
        data_dict['e1i'], data_dict['e2i'], data_dict['e1j'], data_dict['e2j'] = self._collate_entity_idxs(batch)

        data_dict['e1i_count'] = np.stack([data['e1i_count'] for data in batch])
        data_dict['e2i_count'] = np.stack([data['e2i_count'] for data in batch])
        data_dict['e1j_count'] = np.stack([data['e1j_count'] for data in batch])
        data_dict['e2j_count'] = np.stack([data['e2j_count'] for data in batch])
        data_dict['tot_obj_count'] = np.stack([data['tot_obj_count'] for data in batch])
        data_dict['global_obj_ids'] = np.concatenate([data['global_obj_ids'] for data in batch])
        
        data_dict['tot_bow_vec_object_attr_feats'] = tot_bow_vec_object_attr_feats.double()
        data_dict['tot_bow_vec_object_edge_feats'] = tot_bow_vec_object_edge_feats.double()
        data_dict['tot_rel_pose'] = tot_rel_pose.double()
        data_dict['graph_per_obj_count'] = np.stack([data['graph_per_obj_count'] for data in batch])
        data_dict['graph_per_edge_count'] = np.stack([data['graph_per_edge_count'] for data in batch])
        data_dict['edges'] = self._collate_feats(batch, 'edges')
        data_dict['scene_ids'] = np.stack([data['scene_ids'] for data in batch])
        data_dict['obj_ids'] = np.concatenate([data['obj_ids'] for data in batch])
        data_dict['pcl_center'] = np.stack([data['pcl_center'] for data in batch])
        
        data_dict['overlap'] = np.stack([data['overlap'] for data in batch])
        data_dict['batch_size'] = data_dict['overlap'].shape[0]

        return data_dict
        
if __name__ == '__main__':
    from configs import config_scan3r_gt
    cfg = config_scan3r_gt.make_cfg()
    scan3r_ds = Scan3RDataset(cfg, split='val')
    print(len(scan3r_ds))
    scan3r_ds[0]    