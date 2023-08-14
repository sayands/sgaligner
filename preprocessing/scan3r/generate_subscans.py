from tqdm import tqdm
import argparse
import random

import sys
sys.path.append('.')

from configs import config, update_config
from preprocessing.scan3r.subgenscan3r import SubGenScan3R

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='3R Scan configuration file name')
    parser.add_argument('--split', dest='split', default='', type=str, help='split to run on')
    parser.add_argument('--visualise', dest='visualise', action='store_true', help='visualisation of subscene generation - camera trajectory')
    parser.add_argument('--scan_id', dest='scan_id', default='', type=str, help='3RScan scan Id to run subscan generation (only visualisation) on')

    args = parser.parse_args()
    return parser, args

if __name__ == '__main__':
    _, args = parse_args()
    cfg = update_config(config, args.config, ensure_dir=False)
    print('======== Scan3R Subscan Generation with Scene Graphs using config file : {} ========'.format(args.config))
    
    if args.visualise:
        ''' Running subscan generation visualisation '''
        sub_gen_scan3r = SubGenScan3R(cfg, split=args.split)
        scan_id = args.scan_id
        if scan_id == '':
            scan_id = random.choice(sub_gen_scan3r.scan_ids)
            idx = list(sub_gen_scan3r.scan_ids).index(scan_id)
        else:
            idx = list(sub_gen_scan3r.scan_ids).index(scan_id)
        sub_gen_scan3r.logger.info('[INFO] Running Subscan Generation (Only Visualisation) on {}'.format(scan_id))
        sub_gen_scan3r[idx, args.visualise]
    
    else: 
        '''Running subscan generation on data split'''
        sub_gen_scan3r = SubGenScan3R(cfg, split=args.split)
        sub_gen_scan3r.logger.info('[INFO] Running Subscan Generation for {} scans...'.format(args.split))

        for idx in tqdm(range(len(sub_gen_scan3r))):
            sub_gen_scan3r[idx, False]
        
        sub_gen_scan3r.calculate_overlap()
        sub_gen_scan3r.write_metadata()