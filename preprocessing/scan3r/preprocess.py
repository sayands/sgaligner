import os 
import os.path as osp
from tqdm import tqdm
import numpy as np 
from scipy.spatial import ConvexHull
import argparse
import random

import sys
sys.path.append('.')

from utils import define, common, point_cloud, label_mapping
from configs import config, update_config

REL2IDX_SCANNET8 = common.name2idx(osp.join(define.SCAN3R_ORIG_DIR, 'files/scannet8_relationships.txt'))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='3R Scan configuration file name')
    parser.add_argument('--split', dest='split', default='train', type=str, help='split to run subscan generation on')
    parser.add_argument('--remove_nodes', dest='remove_node', default=False,  action='store_true', help='randomly remove nodes from scene graph')      
    parser.add_argument('--remove_edges', dest='remove_edge', default=False,  action='store_true', help='randomly remove edges from scene graph')   
    parser.add_argument('--change_node_semantic', dest='change_node_semantic', default=False,  action='store_true', help='randomly change semantic labels of nodes')        
    parser.add_argument('--change_edge_semantic', dest='change_edge_semantic', default=False,  action='store_true', help='randomly change semantic labels of edges')
    
    args = parser.parse_args()
    if args.remove_node:
        args.mode = 'node_removed'
    if args.remove_edge:
        args.mode = 'edge_removed'
    if args.change_node_semantic:
        args.mode = 'node_semantic_changed'
    if args.change_edge_semantic:
        args.mode = 'edge_semantic_changed'
    else:
        args.mode = 'orig'

    return parser, args

def process_scan(data_dir, rel_data, obj_data, args, cfg, rel2idx):
    scan_id = rel_data['scan']

    if len(rel_data['relationships']) == 0:
        return -1

    objects_ids = [] 
    global_objects_ids = []
    objects_cat = []
    objects_attributes = []
    barry_centers = []

    ply_data = np.load(osp.join(data_dir, 'scenes', scan_id, 'data.npy'))
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
        if not cfg.use_predicted : attribute = [item for sublist in object['attributes'].values() for item in sublist]

        object_id = int(object['id'])
        object_id_for_pcl = int(object['id'])
        
        if args.change_node_semantic and idx in change_obj_indices:
            object_id_for_pcl = np.random.choice(orig_objects_ids)
            while object_id_for_pcl == int(object['id']):
                object_id_for_pcl = np.random.choice(orig_objects_ids)
        
        global_object_id = int(object['global_id'])
        obj_pt_idx = np.where(ply_data['objectId'] == object_id)
        obj_pcl = points[obj_pt_idx]


        if obj_pcl.shape[0] < cfg.preprocess.min_obj_points: continue
        
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
        objects_cat.append(global_object_id)
        if not cfg.use_predicted : objects_attributes.append(attribute)
    
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
        choose_from_rels = [rel for rel in list(rel2idx.keys()) if rel not in ['none', 'inside']]

    for idx, triple in enumerate(relationships):
        sub = int(triple[0])
        obj = int(triple[1])
        rel_id = int(triple[2])  
        rel_name = triple[3]

        if rel_name in list(rel2idx.keys()):
            rel_id = int(rel2idx[rel_name])
            
            if sub in objects_ids and obj in objects_ids:
                if args.change_edge_semantic and idx in rel_change_indices:
                    rel_name_changed = np.random.choice(choose_from_rels)
                    while rel_name == rel_name_changed:
                        rel_name_changed = np.random.choice(choose_from_rels)
                    
                    rel_name = rel_name_changed
                    rel_id = rel2idx[rel_name]
                
                if rel_name == 'inside':
                    assert False
                
                assert rel_id <= len(rel2idx)
                triples.append([sub, obj, rel_id])
                edges_cat.append(rel2idx[rel_name])
                
                if triple[:2] not in pairs:
                    pairs.append([sub, obj])

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
            if i == j or [i, j] in pairs: continue
            triples.append([i, j, rel2idx['none']]) # supplement the 'none' relation
            pairs.append(([i, j]))
            edges_cat.append(rel2idx['none'])
    
    s, o = np.split(np.array(pairs), 2, axis=1)  # All have shape (T, 1)
    s, o = [np.squeeze(x, axis=1) for x in [s, o]]  # Now have shape (T,)

    for index, v in enumerate(s):
        s[index] = object_id2idx[v]  # s_idx
    
    for index, v in enumerate(o):
        o[index] = object_id2idx[v]  # o_idx
    
    edges = np.stack((s, o), axis=1) 

    data_dict = {}
    data_dict['scan_id'] = scan_id
    data_dict['objects_id'] = np.array(objects_ids)
    data_dict['global_objects_id'] = np.array(global_objects_ids)
    data_dict['objects_cat'] = np.array(objects_cat)
    data_dict['triples'] = triples
    data_dict['pairs'] = pairs
    data_dict['edges'] = edges
    data_dict['obj_points'] = object_points
    data_dict['objects_count'] = len(objects_ids)
    data_dict['edges_count'] = len(edges)
    data_dict['object_id2idx'] = object_id2idx
    data_dict['object_attributes'] = objects_attributes
    data_dict['edges_cat'] = edges_cat
    data_dict['rel_trans'] = rel_trans
    data_dict['root_obj_id'] = root_obj_id
    return data_dict

def process_data(args, cfg, rel2idx):
    mode = args.mode
    use_predicted = cfg.use_predicted
    scan_type = cfg.scan_type
    anchor_type_name = cfg.preprocess.anchor_type_name
    out_dirname = '' if scan_type == 'scan' else 'out'
    out_dirname = osp.join(out_dirname, 'predicted') if use_predicted else out_dirname
    data_dir = osp.join(cfg.data.root_dir, out_dirname)
    data_write_dir = osp.join(data_dir, 'files', mode)
    common.ensure_dir(data_write_dir)
    common.ensure_dir(osp.join(data_write_dir, 'data'))
    split = args.split
    
    print('[INFO] Processing subscans from {} split'.format(split))
    
    rel_json_filename = 'relationships.json' if scan_type == 'scan' else 'relationships_subscenes_{}.json'.format(split)
    obj_json_filename = 'objects.json' if scan_type == 'scan' else 'objects_subscenes_{}.json'.format(split)
    scan_ids_filename = '{}_scans.txt'.format(split) if scan_type == 'scan' else '{}_scans_subscenes.txt'.format(split)

    rel_json = common.load_json(osp.join(data_dir, 'files', rel_json_filename))['scans']
    obj_json = common.load_json(osp.join(data_dir, 'files', obj_json_filename))['scans']

    subscan_ids_generated = np.genfromtxt(osp.join(data_dir, 'files', scan_ids_filename), dtype=str)  
    subscan_ids_processed = []

    for subscan_id in tqdm(subscan_ids_generated[:]):
        obj_data = [obj_data for obj_data in obj_json if obj_data['scan'] == subscan_id][0]
        rel_data = [rel_data for rel_data in rel_json if rel_data['scan'] == subscan_id][0]
        data_dict = process_scan(data_dir, rel_data, obj_data, args, cfg, rel2idx)
        
        if type(data_dict) == int: continue

        subscan_ids_processed.append(subscan_id)
        common.write_pkl_data(data_dict, osp.join(data_write_dir, 'data', data_dict['scan_id'] + '.pkl'))
    
    subscan_ids = np.array(subscan_ids_processed) 
    print('[INFO] Updating Overlap Data..')
    anchor_data_filename = osp.join(data_dir, 'files', 'anchors{}_{}.json'.format(anchor_type_name, split))
    raw_anchor_data = common.load_json(anchor_data_filename)
    anchor_data = []
    for anchor_data_idx in tqdm(raw_anchor_data):
        if anchor_data_idx['src'] not in subscan_ids or anchor_data_idx['ref'] not in subscan_ids:
            continue
        anchor_data.append(anchor_data_idx)
    
    common.write_json(anchor_data, osp.join(data_write_dir, 'anchors{}_{}.json'.format(anchor_type_name, split)))

    print('[INFO] Saving {} scan ids...'.format(split))
    print(len(subscan_ids))
    np.savetxt(osp.join(data_write_dir, scan_ids_filename), subscan_ids, fmt='%s')

    return data_dir, data_write_dir, mode

def make_bow_vector(sentence, word_2_idx):
    # create a vector of zeros of vocab size = len(word_to_idx)
    vec = np.zeros(len(word_2_idx))
    for word in sentence:
        if word not in word_2_idx:
            print(word)
            raise ValueError('houston we have a problem')
        else:
            vec[word_2_idx[word]]+=1
    return vec

def calculate_bow_node_edge_feats(data_write_dir, rel2idx):
    print('[INFO] Starting BOW Feature Calculation For Node Edge Features...')
    scan_ids = os.listdir(osp.join(data_write_dir, 'data'))
    scan_ids = sorted([scan_id[:-4] for scan_id in scan_ids])

    idx_2_rel = {idx : relation_name for relation_name, idx in rel2idx.items()}
    
    wordToIx = {}
    for key in rel2idx.keys():
        wordToIx[key] = len(wordToIx)

    print('[INFO] Size of Node Edge Vocabulary - {}'.format(len(wordToIx)))
    print('[INFO] Generated Vocabulary, Calculating BOW Features...')
    for scan_id in scan_ids:
        data_dict_filename = osp.join(data_write_dir, 'data', '{}.pkl'.format(scan_id))
        data_dict = common.load_pkl_data(data_dict_filename)
        
        edge = data_dict['edges']
        objects_ids = data_dict['objects_id']
        triples = data_dict['triples']
        edges = data_dict['edges']

        entities_edge_names = [None] * len(objects_ids)
        for idx in range(len(edges)):
            edge = edges[idx]
            entity_idx = edge[0]
            rel_name = idx_2_rel[triples[idx][2]]

            if rel_name == 'inside':
                print(scan_id)

            if entities_edge_names[entity_idx] is None:
                entities_edge_names[entity_idx] = [rel_name]
            else:
                entities_edge_names[entity_idx].append(rel_name)
            
        entity_edge_feats = None
        for entity_edge_names in entities_edge_names:
            entity_edge_feat = np.expand_dims(make_bow_vector(entity_edge_names, wordToIx), 0)
            entity_edge_feats = entity_edge_feat if entity_edge_feats is None else np.concatenate((entity_edge_feats, entity_edge_feat), axis = 0)

        data_dict['bow_vec_object_edge_feats'] = entity_edge_feats
        assert data_dict['bow_vec_object_edge_feats'].shape[0] == data_dict['objects_count']
        
        common.write_pkl_data(data_dict, data_dict_filename)
    
    print('[INFO] Completed BOW Feature Calculation For Node Edge Features.')

def calculate_bow_node_attr_feats(data_write_dir):
    print('[INFO] Starting BOW Feature Calculation For Node Attribute Features...')
    scan_ids = os.listdir(osp.join(data_write_dir, 'data'))
    scan_ids = sorted([scan_id[:-4] for scan_id in scan_ids])
    
    word_2_ix = common.load_pkl_data(define.OBJ_ATTR_FILENAME)
    for scan_id in tqdm(scan_ids):
        data_dict_filename = osp.join(data_write_dir, 'data', '{}.pkl'.format(scan_id))
        data_dict = common.load_pkl_data(data_dict_filename)
        attributes = data_dict['object_attributes']

        for object_attr in attributes:
            for attr in object_attr:
                if attr not in word_2_ix:
                    word_2_ix[attr] = len(word_2_ix)

    print('[INFO] Size of Node Attribute Vocabulary - {}'.format(len(word_2_ix)))
    print('[INFO] Generated Vocabulary, Calculating BOW Features...')

    for scan_id in scan_ids:
        data_dict_filename = osp.join(data_write_dir, 'data', '{}.pkl'.format(scan_id))
        data_dict = common.load_pkl_data(data_dict_filename)
        attributes = data_dict['object_attributes']

        bow_vec_attrs = None
        for object_attr in attributes:
            bow_vec_attr = np.expand_dims(make_bow_vector(object_attr, word_2_ix), 0)
            bow_vec_attrs = bow_vec_attr if bow_vec_attrs is None else np.concatenate((bow_vec_attrs, bow_vec_attr), axis = 0)
        
        data_dict['bow_vec_object_attr_feats'] = bow_vec_attrs
        assert bow_vec_attrs.shape[0] == data_dict['objects_count']
        common.write_pkl_data(data_dict, data_dict_filename)
    
    print('[INFO] Completed BOW Feature Calculation For Node Attribute Features.')

if __name__ == '__main__':
    _, args = parse_args()
    cfg = update_config(config, args.config, ensure_dir=False)
    random.seed(cfg.seed)
    rel2idx = REL2IDX_SCANNET8
    
    print('======== Scan3R preprocessing with Scene Graphs using config file : {} ========'.format(args.config))
    data_dir, data_write_dir, mode = process_data(args, cfg, rel2idx)

    common.ensure_dir(data_write_dir)
    if not cfg.use_predicted : calculate_bow_node_attr_feats(data_write_dir)
    calculate_bow_node_edge_feats(data_write_dir, rel2idx)