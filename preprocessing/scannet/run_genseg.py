import os
import os.path as osp
import open3d as o3d 
import numpy as np
import argparse
import subprocess
from tqdm import tqdm

import sys
sys.path.append('.')

from utils import common, scannet, util_label
from configs import config, update_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    parser.add_argument('--split', dest='split', default='', type=str, help='split to run on')
    parser.add_argument('--graph_slam_exe', dest='graph_slam_exe', default='', type=str, help='path to graph slam exe file')
    parser.add_argument('--weights_dir', dest='weights_dir', default='', type=str, help='path to traced model dir of 3DSSG')
    args = parser.parse_args()
    return parser, args

def main():
    _, args = parse_args()
    cfg = update_config(config, args.config)

    scannet_path = cfg.data.root_dir
    scannet_scans_path = osp.join(scannet_path, 'scans')
    scannet_files_path = osp.join(scannet_path, 'files')

    scan_ids = scannet.get_scan_ids(scannet_files_path, args.split)

    exe_path = args.graph_slam_exe
    weights_path = args.weights_dir

    for scan_id in tqdm(scan_ids):
        if not osp.exists(osp.join(scannet_scans_path, scan_id)): continue

        scene_in_sens_path = osp.join(scannet_scans_path, scan_id, '{}.sens'.format(scan_id))
        scene_out_dir = osp.join(scannet_scans_path, scan_id)
        cmd = [exe_path, '--pth_in', scene_in_sens_path,  '--pth_out', scene_out_dir, '--pth_model', weights_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(result.stderr)

if __name__ == '__main__':
    main()
        