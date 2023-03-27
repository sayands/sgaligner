from tqdm import tqdm
import argparse
import random

import sys
sys.path.append('.')
from configs import config_scan3r_gt, config_scan3r_pred
from gen_datasets.gen_scan3r import SubGenScan3R

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicted', dest='predicted_sg', action='store_true', default=False, help='run subscan generation with predicted scene graphs')
    parser.add_argument('--visualise', dest='visualise', action='store_true', help='dump visualisation of subscenes')
    parser.add_argument('--scan_id', dest='scan_id', default='', type=str, help='3RScan scan Id to run subscan generation on')
    parser.add_argument('--split', dest='split', default='train', type=str, help='split to run subscan generation on')

    args = parser.parse_args()
    if args.predicted_sg:
        cfg = config_scan3r_pred.make_cfg()
    else:
        cfg = config_scan3r_gt.make_cfg()
    return args, cfg

if __name__ == '__main__':
    args, cfg = parse_args()
    print('======== Scan3R Subscan Generation with {} Truth Scene Graphs ========'.format('GT' if not cfg.predicted_sg else 'Predicted'))
    
    if args.visualise:
        ''' Running subscan generation visualisation '''
        sub_gen_scan3r = SubGenScan3R(cfg, split=args.split)
        scan_id = args.scan_id
        if scan_id == '':
            scan_id = random.choice(sub_gen_scan3r.scan_ids)
            idx = list(sub_gen_scan3r.scan_ids).index(scan_id)
        else:
            idx = list(sub_gen_scan3r.scan_ids).index(scan_id)
        print('[INFO] Running Subscan Generation (Only Visualisation) on {}'.format(scan_id))
        sub_gen_scan3r[idx, args.visualise]
    
    
    else: 
        '''Running subscan generation on data split'''
        sub_gen_scan3r = SubGenScan3R(cfg, split=args.split)
        print('[INFO] Running Subscan Generation for {} scans...'.format(args.split))
        for idx in tqdm(range(len(sub_gen_scan3r))):
            sub_gen_scan3r[idx, False]
        
        sub_gen_scan3r.calculate_overlap()
    


