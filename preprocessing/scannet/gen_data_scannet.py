import os
import os.path as osp
import open3d as o3d 
import numpy as np
import argparse

import sys
sys.path.append('.')

from utils import common, scannet, util_label
from configs import config, update_config

def get_pred_obj_rel(dirname, scan_id, filename, ignore_rels, edge2idx):
    pred_filename = os.path.join(dirname, scan_id, filename)
    preds = common.load_json(pred_filename)[scan_id]

    # Edges 
    relationships = []
    edges = preds['edges']
    for edge in edges.keys():
        sub = edge.split('_')[0]
        obj = edge.split('_')[1]

        edge_probs =  edges[edge]
        best_edge_prob = -1
        best_edge_name = None
        
        for edge_name in edge_probs.keys():
            if abs(edge_probs[edge_name]) > best_edge_prob:
                best_edge_name = edge_name
                best_edge_prob = abs(edge_probs[edge_name])

        if int(sub) is None or int(obj) is None: continue
        if best_edge_name not in ignore_rels and best_edge_name is not None:
            relationships.append([int(sub), int(obj), int(edge2idx[best_edge_name]), best_edge_name])
    
    return {'relationships' : relationships, 'objects' : []}

def process_scan(pth_scan, scan_id, edge2idx, cfg):
    # Params
    label_type = cfg.preprocess.label_type
    filter_segment_size = cfg.preprocess.filter_segment_size
    search_method = cfg.preprocess.search_method
    max_distance =  cfg.preprocess.max_distance
    filter_corr_thres =  cfg.preprocess.filter_corr_thres
    filter_occ_ratio = cfg.preprocess.filter_occ_ratio
    radius_receptive = cfg.preprocess.radius_receptive
    name_same_segment = cfg.preprocess.name_same_segment
    ignore_rels = ['none', name_same_segment]

    pth_prd = osp.join(pth_scan, scan_id, cfg.data.pred_subfix)
    pth_ply = osp.join(pth_scan, scan_id, scan_id+cfg.data.ply_subfix)
    pth_agg = osp.join(pth_scan, scan_id, scan_id+cfg.data.aggre_subfix)
    pth_seg = osp.join(pth_scan, scan_id, scan_id+cfg.data.seg_subfix)

    cloud_gt, points_gt, labels_gt, segments_gt = scannet.load_scannet(pth_ply, pth_agg, pth_seg)
    cloud_pd, points_pd, segments_pd = scannet.load_inseg(pth_prd)
    predictions = get_pred_obj_rel(pth_scan, scan_id, 'predictions.json', ignore_rels, edge2idx)

    # get num of segments
    segment_ids = np.unique(segments_pd) 
    segment_ids = segment_ids[segment_ids!=0]

    segments_pd_filtered=list()
    for seg_id in segment_ids:
        pts = points_pd[np.where(segments_pd==seg_id)]
        if len(pts) > filter_segment_size:
            segments_pd_filtered.append(seg_id)
    segment_ids = segments_pd_filtered

    ''' Check GT segments and labels '''
    label_name_mapping, label_names, label_id_mapping = util_label.getLabelMapping(label_type)
    # Add none to the mapping
    label_name_mapping[len(label_name_mapping) + 1] = 'none'
    label_name_mapping_id2name = { v: k for k, v in label_name_mapping.items()}
    
    instance2labelName = dict()
    size_segments_gt = dict()
    uni_seg_gt_ids = np.unique(segments_gt).tolist()
    
    for seg_id in uni_seg_gt_ids:
        indices = np.where(segments_gt == seg_id)
        seg = segments_gt[indices]
        labels = labels_gt[indices]
        uq_label = np.unique(labels).tolist()
        
        if len(uq_label) > 1:
            max_id=0
            max_value=0
            for id in uq_label:
                if len(labels[labels==id])>max_value:
                    max_value = len(labels[labels==id])
                    max_id = id
            for label in uq_label:
                if label == max_id: continue
                if len(labels[labels==id]) > filter_segment_size: # try to generate new segment
                    new_seg_idx = max(uni_seg_gt_ids)+1
                    uni_seg_gt_ids.append(new_seg_idx)
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = new_seg_idx
                else:    
                    for idx in indices[0]:
                        if labels_gt[idx] == label:
                            segments_gt[idx] = 0
                            labels_gt[idx] = 0 # set other label to 0
            seg = segments_gt[indices]
            labels = labels_gt[indices]
            uq_label = [max_id]
            
        if uq_label[0] == 0 or uq_label[0] > 40:
            name = 'none'
        else:
            name = util_label.NYU40_Label_Names[uq_label[0]-1]
        
        if name not in label_names.values():
            name = 'none'
            
        size_segments_gt[seg_id] = len(seg)
        instance2labelName[seg_id] = name

    size_segments_pd = dict()
    
    ''' Find and count all corresponding segments'''
    tree = o3d.geometry.KDTreeFlann(points_gt.transpose())
    count_seg_pd_2_corresponding_seg_gts = dict() # counts each segment_pd to its corresonding segment_gt
    
    for segment_id in segment_ids:
        segment_indices = np.where(segments_pd == segment_id)[0]
        segment_points = points_pd[segment_indices]        
        
        size_segments_pd[segment_id] = len(segment_points)
        
        if filter_segment_size > 0:
            if size_segments_pd[segment_id] < filter_segment_size:
                continue
            
        for i in range(len(segment_points)):
            point = segment_points[i]
            k, idx, distance = tree.search_knn_vector_3d(point,1)
            if distance[0] > max_distance: continue
            segment_gt = segments_gt[idx][0]
            
            if segment_gt not in instance2labelName: continue
            if instance2labelName[segment_gt] == 'none': continue

            if segment_id not in count_seg_pd_2_corresponding_seg_gts: 
                count_seg_pd_2_corresponding_seg_gts[segment_id] = dict()            
            if segment_gt not in count_seg_pd_2_corresponding_seg_gts[segment_id]: 
                count_seg_pd_2_corresponding_seg_gts[segment_id][segment_gt] = 0
            count_seg_pd_2_corresponding_seg_gts[segment_id][segment_gt] += 1
    
    ''' Find best corresponding segment '''
    map_segment_pd_2_gt = dict() # map segment_pd to segment_gt
    gt_segments_2_pd_segments = dict() # how many segment_pd corresponding to this segment_gt
    for segment_id, cor_counter in count_seg_pd_2_corresponding_seg_gts.items():
        size_pd = size_segments_pd[segment_id]
        max_corr_ratio = -1
        max_corr_seg   = -1
        list_corr_ratio = list()
        for segment_gt, count in cor_counter.items():
            size_gt = size_segments_gt[segment_gt]
            corr_ratio = count/size_pd
            list_corr_ratio.append(corr_ratio)
            if corr_ratio > max_corr_ratio:
                max_corr_ratio = corr_ratio
                max_corr_seg   = segment_gt

        if len(list_corr_ratio ) > 2:
            list_corr_ratio = sorted(list_corr_ratio,reverse=True)
            occ_ratio = list_corr_ratio[1]/list_corr_ratio[0]
        else:
            occ_ratio = 0

        if max_corr_ratio > filter_corr_thres and occ_ratio < filter_occ_ratio:
            '''
            This is to prevent a segment is almost equally occupied two or more gt segments. 
            '''
            map_segment_pd_2_gt[segment_id] = max_corr_seg
            if max_corr_seg not in gt_segments_2_pd_segments:
                gt_segments_2_pd_segments[max_corr_seg] = list()
            gt_segments_2_pd_segments[max_corr_seg].append(segment_id)

    # Get Filtered Relationship Based on predicted segment-ground truth instance correspondences
    pd_segment_ids = []
    filtered_rels = []
    for rel in predictions:
        if rel[0] in pd_segment_ids and rel[1] in pd_segment_ids and rel[3] not in ignore_rels:
            sub = common.get_key_by_value(gt_segments_2_pd_segments, rel[0])
            ob  = common.get_key_by_value(gt_segments_2_pd_segments, rel[1])

            if sub == ob: continue
            if sub is None or ob is None: continue

            row_exists = False
            if len(filtered_rels) > 0: row_exists = np.any(np.all(np.array(filtered_rels) == np.array([sub, ob, rel[2], rel[3]]), axis=1))
            if not row_exists: filtered_rels.append([sub, ob, rel[2], rel[3]])

    relationship_data = {}
    relationship_data['relationships'] = filtered_rels
    relationship_data['scan'] = scan_id

    # Get Objects
    object_data = {}
    object_data['objects'] = []
    object_data['scan'] = scan_id

    for gt_seg_id in uni_seg_gt_ids:
        object_data_idx = { 'label' : instance2labelName[gt_seg_id], 'id' :  gt_seg_id, 
                            'global_obj_id' : label_name_mapping_id2name[instance2labelName[gt_seg_id]]}
        object_data['objects'].append(object_data_idx)
    return relationship_data, object_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    parser.add_argument('--split', dest='split', default='', type=str, help='split to run on')
    args = parser.parse_args()
    return parser, args

def main():
    _, args = parse_args()
    cfg = update_config(config, args.config)

    scannet_path = cfg.data.root_dir
    scannet_scans_path = osp.join(scannet_path, 'scans')
    scannet_files_path = osp.join(scannet_path, 'files')

    scan_ids = scannet.get_scan_ids(scannet_files_path, args.split)
    edge2idx = common.name2idx(os.path.join(scannet_files_path, 'relationships.txt'))
    
    all_rels = dict()
    all_rels['scans'] = []
    all_objs = dict()
    all_objs['scans'] = []
    
    for scan_id in scan_ids:
        if not osp.exists(osp.join(scannet_scans_path, scan_id)): continue
        relationship_data, object_data = process_scan(scannet_scans_path, scan_id, edge2idx, cfg)
        all_rels['scans'].append(relationship_data)
        all_objs['scans'].append(object_data)
    
    common.write_json(all_rels, osp.join(scannet_files_path, 'relationships.json'))
    common.write_json(all_objs, osp.join(scannet_files_path, 'objects.json'))

if __name__ == '__main__':
    main()
    
