import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
import json
from copy import deepcopy
from plyfile import PlyData, PlyElement

import sys
sys.path.append('.')

from utils import common, scan3r, point_cloud, define
from configs import config, update_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    args = parser.parse_args()
    return parser, args

def get_pred_obj_rel(dirname, scan_id, filename, edge2idx, class2idx):
    idx2egde = {idx: edge for edge, idx in edge2idx.items()}
    idx2class = {idx: classname for classname, idx in class2idx.items()}

    pred_filename = os.path.join(dirname, scan_id, filename)
    preds = common.load_json(pred_filename)[scan_id]

    # Edges 
    relationships = []
    pred_edges    = preds['edges']
    
    for edge in pred_edges.keys():
        sub = edge.split('_')[0]
        obj = edge.split('_')[1]

        edge_log_softmax = list(pred_edges[edge].values())
        edge_probs = common.log_softmax_to_probabilities(edge_log_softmax)
        edge_id = np.argmax(edge_probs)
        edge_name = idx2egde[edge_id]
        if edge_name not in ['none'] and edge_name is not None:
            relationships.append([str(sub), str(obj), str(edge_id), edge_name])
    
    # Objects
    objects = []
    object_data = preds['nodes']
    
    for object_id in object_data:
        obj_log_softmax =  list(object_data[object_id].values())
        obj_probs = common.log_softmax_to_probabilities(obj_log_softmax)

        obj_id = np.argmax(obj_probs)
        obj_name = idx2class[obj_id]
        
        if obj_name not in ['none'] and obj_name is not None: 
            objects.append({'label' : obj_name, 'id' : str(object_id), 'global_id' : str(obj_id)})
    
    return {'relationships' : relationships, 'objects' : objects}

def filter_merge_same_part_pc(segments_pd, objects, relationships):
    instanceid2objname = {int(object_data['id']) : object_data['label'] for object_data in objects}

    pairs = []
    filtered_relationships = []
    for relationship in relationships:
        if relationship[-1] != define.NAME_SAME_PART : 
            filtered_relationships.append(relationship)
        elif relationship[-1] == define.NAME_SAME_PART and instanceid2objname[int(relationship[0])] == instanceid2objname[int(relationship[1])]:
            pairs.append([int(relationship[0]), int(relationship[1])])
            filtered_relationships.append(relationship)

    same_parts = common.merge_duplets(pairs)
    relationship_data = deepcopy(filtered_relationships)

    del_objects_idxs = []

    for same_part in same_parts:
        root_segment_id = same_part[0]
        
        for part_segment_id in same_part[1:]:
            segments_pd[np.where(segments_pd == part_segment_id)[0]] = root_segment_id
        
            for idx, object_data_raw in enumerate(objects[:]):
                if int(object_data_raw['id']) == part_segment_id: del_objects_idxs.append(idx)

            for idx, (sub, ob, rel_id, rel_name) in enumerate(filtered_relationships):
                sub = int(sub)
                ob = int(ob)
                rel_id = int(rel_id)

                
                if sub == part_segment_id: sub = root_segment_id
                if ob == part_segment_id: ob = root_segment_id

                if sub == ob: continue
                
                relationship_data[idx][0] = str(sub)
                relationship_data[idx][1] = str(ob)
    
    del_objects_idxs = list(set(del_objects_idxs))
    object_data = [object_data_idx for idx, object_data_idx in enumerate(objects) if idx not in del_objects_idxs]
    segment_ids_filtered = np.unique(segments_pd)
    
    return segment_ids_filtered, segments_pd, object_data, relationship_data

def process_scan(scan3r_scans_path, scan_id, edge2idx, class2idx, cfg):
    filter_segment_size = cfg.preprocess.filter_segment_size
        
    pth_prd = osp.join(scan3r_scans_path, scan_id, cfg.data.pred_subfix)
    cloud_pd, points_pd, segments_pd = point_cloud.load_inseg(pth_prd)

    # get num of segments
    segment_ids = np.unique(segments_pd) 
    segment_ids = segment_ids[segment_ids!=0]

    segments_pd_filtered=list()
    for seg_id in segment_ids:
        pts = points_pd[np.where(segments_pd==seg_id)]
        if len(pts) > filter_segment_size: segments_pd_filtered.append(seg_id)
    
    segment_ids = segments_pd_filtered
    rel_obj_data_dict = get_pred_obj_rel(scan3r_scans_path, scan_id, 'predictions.json', edge2idx, class2idx)
    
    relationships = []
    objects = []
    filtered_segments_ids = []
    
    for object_data in rel_obj_data_dict['objects']:
        if int(object_data['id']) in segment_ids: filtered_segments_ids.append(int(object_data['id']))
    
    segment_ids = filtered_segments_ids

    for rel in rel_obj_data_dict['relationships']:
        if int(rel[0]) in segment_ids and int(rel[1]) in segment_ids:
            relationships.append(rel)
    
    for seg_id in segment_ids:
        obj_data = [object_data for object_data in rel_obj_data_dict['objects'] if seg_id == int(object_data['id'])]
        if len(obj_data) == 0: continue
        objects.append(obj_data[0])

    assert len(segment_ids) == len([object_data['id'] for object_data in objects])

    points_pd_mask = np.isin(segments_pd, segment_ids)
    points_pd = points_pd[np.where(points_pd_mask == True)[0]]
    segments_pd = segments_pd[np.where(points_pd_mask == True)[0]]

    segment_ids, segments_pd, objects, relationships = filter_merge_same_part_pc(segments_pd, objects, relationships)
    assert len(segment_ids) == len([object_data['id'] for object_data in objects])

    relationship_data = {'relationships' : relationships, 'scan' : scan_id}
    object_data = {'objects' : objects, 'scan': scan_id}

    # Filter segments
    segments_ids_pc_mask = np.isin(segments_pd, segment_ids)
    points_pd = points_pd[np.where(segments_ids_pc_mask == True)[0]]
    segments_pd = segments_pd[np.where(segments_ids_pc_mask == True)[0]]

    # Create ply file
    verts = []
    for idx, v in enumerate(points_pd):
        vert = (v[0], v[1], v[2], segments_pd[idx])
        verts.append(vert)
    
    verts = np.asarray(verts, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('label', 'u2')])
    plydata = PlyData([PlyElement.describe(verts, 'vertex', comments=['vertices'])])
    savename = osp.join(scan3r_scans_path, scan_id, 'inseg_filtered.ply')
    with open(savename, mode='wb') as f: PlyData(plydata).write(f)
    
    return relationship_data, object_data

def main():
    print('==== 3RSCAN Data Generation ====')
    _, args = parse_args()
    cfg = update_config(config, args.config)
    message = json.dumps(cfg, indent=4)
    print(message)
    
    scan3r_path = cfg.data.root_dir
    scan3r_scans_path = osp.join(scan3r_path, 'scenes')
    scan3r_files_path = osp.join(scan3r_path, 'files')
    scan3r_predicted_files_path = osp.join(scan3r_path, 'files/predicted')
    common.ensure_dir(scan3r_predicted_files_path)

    splits = ['train', 'val']
    all_rels = dict()
    all_rels['scans'] = []
    all_objs = dict()
    all_objs['scans'] = []

    for split in splits:
        print('[INFO]Started on {} split.'.format(split))
        scan_ids  = scan3r.get_scan_ids(scan3r_files_path, split)
        edge2idx  = common.name2idx(os.path.join(scan3r_files_path, 'scannet8_relationships.txt'))
        class2idx = common.name2idx(os.path.join(scan3r_files_path, 'scannet20_classes.txt'))

        scan_ids_pred = []

        for scan_id in tqdm(scan_ids[:]):
            if not osp.exists(osp.join(scan3r_scans_path, scan_id, cfg.data.pred_subfix)): continue
            
            relationship_data, object_data = process_scan(scan3r_scans_path, scan_id, edge2idx, class2idx, cfg)
            all_rels['scans'].append(relationship_data)
            all_objs['scans'].append(object_data)
            scan_ids_pred.append(scan_id)
        

        print('[INFO]Finished {} split.'.format(split))

        scan_ids_pred = np.array(scan_ids_pred)
        np.savetxt(osp.join(scan3r_predicted_files_path, '{}_scans.txt'.format(split)), scan_ids_pred, fmt='%s')

    common.write_json(all_rels, osp.join(scan3r_predicted_files_path, 'relationships.json'))
    common.write_json(all_objs, osp.join(scan3r_predicted_files_path, 'objects.json'))

if __name__ == '__main__':
    main()