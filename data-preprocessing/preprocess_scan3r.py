import os 
import os.path as osp
from tqdm import tqdm
import numpy as np 
from scipy.spatial import ConvexHull
import argparse

import sys
sys.path.append('.')

from utils import define, common, point_cloud, scan3r, label_mapping
from configs import config_scan3r_gt, config_scan3r_pred

CLASS2IDX_SCAN3R = label_mapping.get_class_2_idx(osp.join(define.SCAN3R_ORIG_DIR, 'files/classes.txt'))
REL2IDX_SCAN3R = label_mapping.get_class_2_idx(osp.join(define.SCAN3R_ORIG_DIR, 'files/relationships.txt'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted', dest='predicted_sg', action='store_true', default=False, help='run subscan generation with predicted scene graphs')
    parser.add_argument('--split', dest='split', default='train', type=str, help='split to run subscan generation on')
    parser.add_argument('--remove_nodes', dest='remove_node', default=False,  action='store_true', help='randomly remove nodes from scene graph')      
    parser.add_argument('--remove_edges', dest='remove_edge', default=False,  action='store_true', help='randomly remove edges from scene graph')   
    parser.add_argument('--change_node_semantic', dest='change_node_semantic', default=False,  action='store_true', help='randomly change semantic labels of nodes')        
    parser.add_argument('--change_edge_semantic', dest='change_edge_semantic', default=False,  action='store_true', help='randomly change semantic labels of edges')
    
    args = parser.parse_args()
    if args.predicted_sg:
        cfg = config_scan3r_pred.make_cfg()
    else:
        cfg = config_scan3r_gt.make_cfg()

    if args.remove_node:
        args.mode = 'node_removed'
    if args.remove_edge:
        args.mode = 'edge_removed'
    if args.change_node_semantic:
        args.mode = 'node_semantic_changed'
    if args.change_edge_semantic:
        args.mode = 'edge_semantic_changed'

    return args, cfg

def process_scan(rel_data, obj_data, args, cfg):
    scan_id = rel_data['scan']

    if len(rel_data['relationships']) == 0:
        return -1

    objects_ids = [] 
    global_objects_ids = []
    objects_cat = []
    objects_attributes = []
    barry_centers = []

    ply_data = np.load(osp.join(cfg.data_dir, 'out/scenes', scan_id, 'data.npy'))
    points = np.stack([ply_data['x'], ply_data['y'], ply_data['z']]).transpose((1, 0))

    object_points = {}
    for pc_resolution in cfg.preprocess.pc_resolutions:
        object_points[pc_resolution] = []

    object_data = obj_data['objects'] 

    if args.remove_node:
        num_obj_to_keep = int(((100 - np.random.randint(15, 41)) / 100.0) * len(object_data))
        keep_obj_indices = np.random.choice(len(object_data), num_obj_to_keep, replace=False)
        object_data = [object_data[idx] for idx in keep_obj_indices]
    
    if args.change_node_semantic:
        num_obj_to_change = int((np.random.randint(15, 41) / 100.0) * len(object_data))
        change_obj_indices = np.random.choice(len(object_data), num_obj_to_change, replace=False)
        orig_objects_ids = []

        for idx, object in enumerate(object_data):
            orig_objects_ids.append(int(object['id'])) 
        
    for idx, object in enumerate(object_data):
        attribute = [item for sublist in object['attributes'].values() for item in sublist]

        object_id = int(object['id'])
        object_id_for_pcl = int(object['id'])
        
        if args.change_node_semantic and idx in change_obj_indices:
            object_id_for_pcl = np.random.choice(orig_objects_ids)
            while object_id_for_pcl == int(object['id']):
                object_id_for_pcl = np.random.choice(orig_objects_ids)
        
        global_object_id = int(object['global_id'])
        obj_pt_idx = np.where(ply_data['objectId'] == global_object_id)
        obj_pcl = points[obj_pt_idx]

        if obj_pcl.shape[0] < cfg.data.min_obj_points: continue
        
        hull = ConvexHull(obj_pcl)
        cx = np.mean(hull.points[hull.vertices,0])
        cy = np.mean(hull.points[hull.vertices,1])
        cz = np.mean(hull.points[hull.vertices,2])

        for pc_resolution in object_points.keys():
            obj_pcl = point_cloud.pcl_farthest_sample(obj_pcl, pc_resolution)
            object_points[pc_resolution].append(obj_pcl)
        
        barry_centers.append([cx, cy, cz])
        objects_ids.append(object_id)
        global_objects_ids.append(global_object_id)
        objects_cat.append(CLASS2IDX_SCAN3R[object['label']])
        objects_attributes.append(attribute)
    
    for pc_resolution in object_points.keys():
        object_points[pc_resolution] = np.array(object_points[pc_resolution])
    
    if len(objects_ids) < 2:
        return -1
    
    object_id2idx = {}  # convert object id to the index in the tensor
    for index, v in enumerate(objects_ids):
        object_id2idx[v] = index

    relationships = rel_data['relationships']
    triples = []
    pairs = []
    edges_cat = []

    if args.remove_edge:
        num_rel_to_keep = int(((100 - np.random.randint(15, 41)) / 100.0) * len(relationships))
        keep_indices = np.random.choice(len(relationships), num_rel_to_keep, replace=False)
        relationships = [relationships[idx] for idx in keep_indices]

    if args.change_edge_semantic:
        num_rel_to_change = int((np.random.randint(15, 41) / 100.0) * len(relationships))
        rel_change_indices = np.random.choice(len(relationships), num_rel_to_change, replace=False)
        choose_from_rels = [rel for rel in list(REL2IDX_SCAN3R.keys()) if rel not in ['none', 'inside']]

    for idx, triple in enumerate(relationships):
        if triple[0] in objects_ids and triple[1] in objects_ids:
            sub = triple[0]
            obj = triple[1]
            rel_id = triple[2]    
            rel_name = triple[3]

            if args.change_edge_semantic and idx in rel_change_indices:
                rel_name_changed = np.random.choice(choose_from_rels)
                while rel_name == rel_name_changed:
                    rel_name_changed = np.random.choice(choose_from_rels)
                
                rel_name = rel_name_changed
                rel_id = REL2IDX_SCAN3R[rel_name]
            
            if rel_name == 'inside':
                assert False

            triples.append([sub, obj, rel_id])
            edges_cat.append(REL2IDX_SCAN3R[rel_name])
            
            if triple[:2] not in pairs:
                pairs.append(triple[:2])

    if len(pairs) == 0:
        return -1

    # Root Object - object with highest outgoing degree
    all_edge_objects_ids = np.array(pairs).flatten()
    root_obj_id = np.argmax(np.bincount(all_edge_objects_ids))
    root_obj_idx = object_id2idx[root_obj_id]

    # Calculate barry center and relative translation
    rel_trans = []
    for barry_center in barry_centers:
        rel_trans.append(np.subtract(barry_centers[root_obj_idx], barry_center))
    
    rel_trans = np.array(rel_trans)
    
    for i in objects_ids:
        for j in objects_ids:
            if i == j or [i, j] in pairs: 
                continue
            triples.append([i, j, REL2IDX_SCAN3R['none']]) # supplement the 'none' relation
            pairs.append(([i, j]))
            edges_cat.append(REL2IDX_SCAN3R['none'])
    
    s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
    s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)

    for index, v in enumerate(s):
        s[index] = object_id2idx[v]  # s_idx
    for index, v in enumerate(o):
        o[index] = object_id2idx[v]  # o_idx
    edges = np.stack((s, o), axis=1) 

    dataDict = {}
    dataDict['scan_id'] = scan_id
    dataDict['objects_id'] = np.array(objects_ids)
    dataDict['global_objects_id'] = np.array(global_objects_ids)
    dataDict['objects_cat'] = np.array(objects_cat)
    dataDict['triples'] = triples
    dataDict['pairs'] = pairs
    dataDict['edges'] = edges
    dataDict['obj_points'] = object_points
    dataDict['objects_count'] = len(objects_ids)
    dataDict['edges_count'] = len(edges)
    dataDict['object_id2idx'] = object_id2idx
    dataDict['object_attributes'] = objects_attributes
    dataDict['edges_cat'] = edges_cat
    dataDict['rel_trans'] = rel_trans
    dataDict['root_obj_id'] = root_obj_id

    return dataDict

def process_data(args, cfg):
    mode = args.mode
    data_dir = osp.join(cfg.data_dir, 'out')
    data_write_dir = osp.join(data_dir, 'files', mode)
    common.ensure_dir(data_write_dir)
    split = args.split

    rel_json = common.load_json(osp.join(data_dir, 'files', 'relationships_subscenes_{}.json'.format(split)))['scans']
    obj_json = common.load_json(osp.join(data_dir, 'files', 'objects_subscenes_{}.json'.format(split)))['scans']

    subscan_ids_generated = np.genfromtxt(osp.join(data_dir, 'files', '{}_scans_subscenes.txt'.format(split)), dtype=str)    
    subscan_ids_processed = []

    for subscan_id in tqdm(subscan_ids_generated):
        obj_data = [obj_data for obj_data in obj_json if obj_data['scan'] == subscan_id][0]
        rel_data = [rel_data for rel_data in rel_json if rel_data['scan'] == subscan_id][0]
        data_dict = process_scan(rel_data, obj_data, split, mode)
        
        if type(data_dict) == int: continue
        subscan_ids_processed.append(subscan_id)
        
        common.write_pkl_data(data_dict, osp.join(data_write_dir, data_dict['scan_id'] + '.pkl'))
    
    subscan_ids = np.array(subscan_ids) 
    print('[INFO] Updating Overlap Data..')
    anchor_data_filename = osp.join(data_dir, 'files', 'anchors_{}.json'.format(split))
    raw_anchor_data = common.load_json(anchor_data_filename)
    anchor_data = []
    for anchor_data_idx in tqdm(raw_anchor_data):
        if anchor_data_idx['src'] not in subscan_ids or anchor_data_idx['ref'] not in subscan_ids:
            continue
        anchor_data.append(anchor_data_idx)
    
    common.write_json(anchor_data, osp.join(data_write_dir, 'anchors_{}.json'.format(split)))

    print('[INFO] Saving {} scan ids...'.format(split))
    np.savetxt(osp.join(data_write_dir, '{}_scans_subscenes.txt'.format(split)), subscan_ids, fmt='%s')

if __name__ == '__main__':
    args, cfg = parse_args()
    print('======== Scan3R Subscan preprocessing with {} Scene Graphs ========'.format('GT' if not cfg.predicted_sg else 'Predicted'))


