import os.path as osp
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

import open3d as o3d
from torch.utils.data import Dataset

from utils import common, point_cloud, scan3r, open3d, visualisation
from utils.logger import Logger

class SubGenScan3R(Dataset):
    ''' Subset Generation from 3RScan dataset '''
    def __init__(self, cfg, split='train'):
        self.predicted = cfg.use_predicted
        self.scene_dir = osp.join(cfg.data.root_dir, 'scenes')
        
        file_dirname = 'files/predicted' if self.predicted else 'files'
        self.file_dir = osp.join(cfg.data.root_dir, file_dirname) 
        
        out_dirname = 'out/predicted' if self.predicted else 'out' 
        self.out_dir = osp.join(cfg.data.root_dir, out_dirname)
        self.scene_out_dir = osp.join(self.out_dir, 'scenes')
        self.file_out_dir = osp.join(self.out_dir, 'files')
        self.split = split

        common.ensure_dir(self.scene_out_dir)
        common.ensure_dir(self.file_out_dir)
        random.seed(cfg.seed)

        self.logger = Logger(log_file=osp.join(self.file_out_dir, 'log.txt'))
        
        self.scan_ids = scan3r.get_scan_ids(self.file_dir, self.split)

        # Load all relationships
        self.scan_rels = common.load_json(osp.join(self.file_dir, 'relationships.json'))['scans']

        # Load objects json
        self.scan_objs = common.load_json(osp.join(self.file_dir, 'objects.json'))['scans']

        self.num_subscans_per_scan = cfg.preprocess.subscenes_per_scene
        self.subscene_rels = {'scans' : []}
        self.subscene_objs = {'scans' : []}

        self.obj_pt_scene_thresh = cfg.preprocess.min_obj_points
        self.logger.info('[INFO] Loaded {} {} scan data...'.format(self.__len__(), self.split))
        
        self.label_file_name = 'labels.instances.align.annotated.v2.ply' if not self.predicted else 'inseg_filtered.ply'
        self.save_name = 'data.npy'
        self.skip = None if not self.predicted else 5

    def gen_scene_graph(self, scan_id, idx, ply_data, visible_pts_mask):
        obj_json_scan = [scan_obj for scan_obj in self.scan_objs if scan_obj['scan'] == scan_id][0]['objects']

        subscan_id = '{}_{}'.format(scan_id, idx)
        visible_pts_idx = np.where(visible_pts_mask == True)[0]
        
        # Get visible points, colors and object IDs
        visible_pcl_data, visible_pts_obj_ids = scan3r.create_ply_data(ply_data, visible_pts_idx) if not self.predicted \
                                                else scan3r.create_ply_data_predicted(ply_data, visible_pts_idx)
        unique_visible_pts_obj_ids = np.unique(visible_pts_obj_ids)

        subscan_obj = [scan_obj for scan_obj in obj_json_scan if int(scan_obj['id']) in unique_visible_pts_obj_ids]
        subscan_obj_data = {'scan' : subscan_id, 'objects' : subscan_obj}

        # Append to create sub scene objects.json
        self.subscene_objs['scans'].append(subscan_obj_data)

        # Get the scan relationship JSON
        scan_rels = [item for item in self.scan_rels if item['scan'] == scan_id][0]
        scan_rels = scan_rels['relationships']
        
        # Loop through scan relationships
        subscan_rels = []
        for (sub_id, ob_id, rel_id, rel_name) in scan_rels:
            num_sub_pts = len(np.where(visible_pts_obj_ids == int(sub_id))[0])
            num_ob_pts = len(np.where(visible_pts_obj_ids == int(ob_id))[0])

            if num_sub_pts > self.obj_pt_scene_thresh and num_ob_pts > self.obj_pt_scene_thresh:
                subscan_rels.append([sub_id, ob_id, rel_id, rel_name])
        
        subscan_rel_data = {'relationships' : subscan_rels, "scan" : subscan_id}
        # Append to create sub scene relationships.json
        self.subscene_rels['scans'].append(subscan_rel_data)
        subscan_data = {'pcl' : visible_pcl_data, 'subscan_id' : subscan_id, 'relationships' : subscan_rel_data, 'objects' : subscan_obj_data}

        return subscan_data

    def __len__(self):
        return self.scan_ids.shape[0]

    def calculate_overlap(self):
        self.logger.info('[INFO] Calculating Overlap on Generated subscans...')
        
        anchor_file_name = osp.join(self.file_out_dir , 'anchors_{}_all.json'.format(self.split))
        all_subscan_ids = os.listdir(self.scene_out_dir)
        overlap_data = []

        for scan_id in tqdm(self.scan_ids):
            subscan_ids = [subscan for subscan in all_subscan_ids if subscan.startswith(scan_id)]
            subscan_idxs = range(0, len(subscan_ids))
            subscan_idx_pairs = list(itertools.combinations(subscan_idxs, 2))
        
            subscan_ply_data_all = []
            for idx in range(0, len(subscan_ids)):
                _, subscan_ply_data = scan3r.load_plydata_npy(osp.join(self.scene_out_dir, subscan_ids[idx], 'data.npy'), obj_ids=None, return_ply_data=True)
                subscan_ply_data_all.append(subscan_ply_data)
            
            for subscan_pair in subscan_idx_pairs:
                src_ply_data = subscan_ply_data_all[subscan_pair[0]]
                ref_ply_data = subscan_ply_data_all[subscan_pair[1]]

                src_points = np.stack([src_ply_data['x'], src_ply_data['y'], src_ply_data['z']]).transpose((1, 0))
                ref_points = np.stack([ref_ply_data['x'], ref_ply_data['y'], ref_ply_data['z']]).transpose((1, 0))
                overlap_ratio, common_pts_idx_src = point_cloud.compute_pcl_overlap(src_points, ref_points)

                if overlap_ratio >= 0.1 and overlap_ratio <= 0.9:
                    anchor_obj_ids = np.unique(src_ply_data['objectId'][common_pts_idx_src])
                    overlap_data.append({'src' : subscan_ids[subscan_pair[0]], 'ref' : subscan_ids[subscan_pair[1]], 
                            'overlap' : overlap_ratio, 'anchorIds' : anchor_obj_ids.tolist()})
        
        common.write_json(overlap_data, anchor_file_name)

    def write_metadata(self):
        self.logger.info('[INFO] Writing Relationships + Objects Data...')
        common.write_json(self.subscene_rels, osp.join(self.file_out_dir, 'relationships_subscenes_{}.json'.format(self.split)))
        common.write_json(self.subscene_objs, osp.join(self.file_out_dir,'objects_subscenes_{}.json'.format(self.split)))
        
        self.logger.info('[INFO] Choosing a (realistic) subset of subscenes generated...')
        all_subscan_ids = [subscan_id for subscan_id in os.listdir(self.scene_out_dir) if subscan_id[:subscan_id.index('_')] in self.scan_ids]
        all_subscan_ids = np.array(all_subscan_ids)
        self.logger.info('[INFO] Displaying information for {} split...'.format(self.split))
        self.logger.info('[INFO] Total 3RScan scenes : {}'.format(self.scan_ids.shape[0]))
        self.logger.info('[INFO] Total generated subscenes : {}'.format(all_subscan_ids.shape[0]))

        subscan_ids = []
        for scan_id in self.scan_ids:
            subscan_ids_scan = [subscan_id for subscan_id in all_subscan_ids if subscan_id.startswith(scan_id)]

            if len(subscan_ids_scan) > self.num_subscans_per_scan:
                subscan_ids_scan = np.random.choice(subscan_ids_scan, self.num_subscans_per_scan, replace=False)
            
            subscan_ids.append(subscan_ids_scan)
        
        subscan_ids = np.concatenate(subscan_ids)
        self.logger.info('[INFO] Chosen subscenes : {}'.format(subscan_ids.shape[0]))

        anchor_data_dumped = common.load_json(osp.join(self.file_out_dir, 'anchors_{}_all.json'.format(self.split)))
        self.logger.info('[INFO] Total generated no.of pairs - {}'.format(len(anchor_data_dumped)))

        anchor_data = []
        for anchor_data_idx in anchor_data_dumped:
            if anchor_data_idx['src'] in subscan_ids and anchor_data_idx['ref'] in subscan_ids:
                anchor_data.append(anchor_data_idx)
        
        self.logger.info('[INFO] Chosen no.of pairs - {}'.format(len(anchor_data)))

        np.savetxt(osp.join(self.file_out_dir, '{}_scans_subscenes.txt'.format(self.split)), subscan_ids, fmt='%s')
        common.write_json(anchor_data, osp.join(self.file_out_dir, 'anchors_{}.json'.format(self.split)))
    
    def __getitem__(self, data):
        idx = data[0]
        visualise =  data[1]

        if visualise :
            vis = open3d.make_open3d_visualiser()
        
        scan_id = self.scan_ids[idx]
        frame_idxs = scan3r.load_frame_idxs(self.scene_dir, scan_id, skip=self.skip)

        # Load all frame poses
        frame_poses = scan3r.load_all_poses(self.scene_dir, scan_id, frame_idxs)

        # Load scene pcl
        ply_data = scan3r.load_ply_data(self.scene_dir, scan_id, self.label_file_name)
        scene_pts = np.stack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z'])).transpose()
        if scene_pts.shape[0] == 0: return
        
        scene_pcd = open3d.make_open3d_point_cloud(visualisation.remove_ceiling(scene_pts))

        if visualise: 
            scene_pcd.paint_uniform_color(visualisation.get_random_color() / 255.)
            vis.add_geometry(scene_pcd)
        
        cam_centers = scan3r.find_cam_centers(frame_idxs, frame_poses)

        # Load intrinsic information and generate maximum points per subscene 
        intrinsic_info = scan3r.load_intrinsics(self.scene_dir, scan_id)
        max_pts_subscan = random.randint(int(0.2 * scene_pts.shape[0]), int(0.5 * scene_pts.shape[0]))    
        curr_visible_mask = np.zeros(scene_pts.shape[0]).astype('bool')
        
        frame_cnt = 0
        subscan_idx = 0

        start_idx = 0
        # Loop through all frames
        while frame_cnt < len(frame_idxs):
            frame_pose = frame_poses[frame_cnt]

            # get the points from scene PLY visible in current frame
            frame_visible_mask = point_cloud.get_visible_pts_from_cam_pose(scene_pts, frame_pose, intrinsic_info)
            
            # get total no.of points visible till given idx 
            curr_visible_mask = np.logical_or(frame_visible_mask, curr_visible_mask)
            subscan_pts = scene_pts[curr_visible_mask]

            # if maxPtsPerSubscene reached, add to subscene list and re-start from next frame ID
            if subscan_pts.shape[0] >= max_pts_subscan:
                if visualise:
                    # TODO - add subscene visualisation
                    # subscan_pcd = open3d.make_open3d_point_cloud(subscan_pts)
                    # subscan_pcd.paint_uniform_color(visualisation.get_random_color() / 255.)
                    # open3d.draw_geometries(subscan_pcd)

                    ''' Camera trajectory visualisation '''
                    end_idx = frame_cnt
                    cam_centers_subscan = cam_centers[start_idx : end_idx]
                    start_idx = end_idx
                    for i in range(cam_centers_subscan.shape[0]):
                        sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                        sphere_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(sphere_mesh.vertices) + cam_centers_subscan[i])
                        color = visualisation.get_random_color() / 255.
                        sphere_mesh.vertex_colors = o3d.utility.Vector3dVector(np.zeros_like(np.asarray(sphere_mesh.vertices)) + color)
                        vis.add_geometry(sphere_mesh)

                else:
                    subscan_data = self.gen_scene_graph(scan_id, subscan_idx, ply_data, curr_visible_mask)
                    subscan_out_dir = osp.join(self.scene_out_dir, subscan_data['subscan_id'])
                    common.ensure_dir(subscan_out_dir)
                    np.save(osp.join(subscan_out_dir, self.save_name), subscan_data['pcl'])

                subscan_idx += 1
                curr_visible_mask = np.zeros(scene_pts.shape[0]).astype('bool')
            
            frame_cnt += 1

        if visualise:
            self.logger.info('[INFO] Generated {} subscans...'.format(subscan_idx))
            vis.run()