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

def gen_rescan_to_refscan_map(mode, split):
    scans_dir = define.SCAN3R_ORIG_DIR
    scans_files_dir = osp.join(scans_dir, 'files')
    all_scan_data = common.load_json(osp.join(scans_files_dir, '3RScan.json'))
    
    scan_ids = np.genfromtxt(osp.join(scans_files_dir, '{}_scans.txt'.format(split)), dtype=str)
    anchor_data = []
    
    for scan_data in all_scan_data:
        ref_scan_id = scan_data['reference']

        rescan_ids = [scan['reference'] for scan in scan_data['scans']]

        for rescan_id in rescan_ids:
            if rescan_id in scan_ids and ref_scan_id in scan_ids:
                anchor_data.append({ 'src' : rescan_id, 'ref' : ref_scan_id})
    
    print('[INFO] Processed {} {} rescan-to-scan...'.format(len(anchor_data), split))

    common.write_json(anchor_data, osp.join(scans_files_dir, 'anchors_rescans_to_refscans_{}.json'.format(split)))

def main():
    args = parse_args()
    gen_rescan_to_refscan_map(args.mode, args.split) 

if __name__ == '__main__':
    main()
