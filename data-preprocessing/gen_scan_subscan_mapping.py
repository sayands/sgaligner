import os.path as osp
import numpy as np 
import random
import argparse

import sys
sys.path.append('.')
from utils import common, define

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', dest='split', default='train', type=str, help='split to generate scan-subscan mapping on')
    parser.add_argument('--mode', dest='mode', default='orig', type=str, help='the data mode to generate scan-subscan mapping with')
    args = parser.parse_args()
    return args

def gen_scan_subscan_mapping(mode, split):
    scan_ids = np.genfromtxt(osp.join(define.SCAN3R_ORIG_DIR, 'files', '{}_scans.txt'.format(split)), dtype=str)
    random.shuffle(scan_ids)
    subscan_ids = np.genfromtxt(osp.join(define.SCAN3R_SUBSCENES_DIR, 'files/{}/{}_scans_subscenes.txt'.format(mode, split)), dtype=str)
    scan_subscan_map = {}
    for scan_id in scan_ids:
        subscan_ids_per_scan_id = [subscan_id for subscan_id in subscan_ids if subscan_id.startswith(scan_id)]
        subscan_ids_per_scan_id = sorted(subscan_ids_per_scan_id)
        if len(subscan_ids_per_scan_id) == 0: continue
        scan_subscan_map[scan_id] = subscan_ids_per_scan_id

    print('[INFO] Proccessed {} 3RScan {} scans.'.format(len(scan_ids), split))
    return scan_subscan_map

def main():
    args = parse_args()
    scan_subscan_map = gen_scan_subscan_mapping(args.mode, args.split)
    common.write_json(scan_subscan_map, osp.join(define.SCAN3R_SUBSCENES_DIR, 'files', 'orig/scan_subscan_map_{}.json'.format(args.split)))

if __name__ == '__main__':
    main()