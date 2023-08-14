import os
import os.path as osp
import numpy as np 
import itertools
from tqdm import tqdm
import random

import sys
sys.path.append('.')
from utils import common, define

def gen_fileset(subscans_files_dir, split, mode='orig'):
    subscan_ids = np.genfromtxt(osp.join(subscans_files_dir, mode, '{}_scans_subscenes.txt'.format(split)), dtype=str)
    random.shuffle(subscan_ids)
    all_overlap_data = common.load_json(osp.join(subscans_files_dir, 'anchors_{}_all.json'.format(split)))

    subscan_ids_indices = range(0, len(subscan_ids))
    subscan_ids_indices_pairs = list(itertools.combinations(subscan_ids_indices, 2))
    random.shuffle(subscan_ids_indices_pairs)

    overlap_data_dumped = common.load_json(osp.join(subscans_files_dir, mode, 'anchors_{}.json'.format(split)))
    anchor_data = []
    count = 0
    for subscan_ids_pair in tqdm(subscan_ids_indices_pairs):
        src_scan_id = subscan_ids[subscan_ids_pair[0]]
        tgt_scan_id = subscan_ids[subscan_ids_pair[1]]

        found_flag = False
        for overlap_data_idx in all_overlap_data:
            if overlap_data_idx['src'] == src_scan_id and overlap_data_idx['ref'] == tgt_scan_id:
                found_flag = True
                break
        
        if count < len(overlap_data_dumped):
            if not found_flag:
                anchor_data.append({'src' : src_scan_id, 'ref' : tgt_scan_id, 'overlap' : 0.0, 'anchorIds' : []})
                count += 1
        else: break
    
    anchor_data = np.concatenate([overlap_data_dumped, anchor_data])
    anchor_data = list(anchor_data)
    random.shuffle(anchor_data)

    print('[INFO] Generated {} {} subscan pairs overlap + without overlap ...'.format(len(anchor_data), split))
    common.write_json(anchor_data, osp.join(subscans_files_dir, mode, 'anchors_subscan_anchors_w_wo_overlap_{}.json'.format(split)))

def main():
    random.seed(42)
    subscans_dir = define.SCAN3R_SUBSCENES_DIR
    subscans_files_dir = osp.join(subscans_dir, 'files')
    split = 'val'

    gen_fileset(subscans_files_dir, split)

if __name__ == '__main__':
    main()