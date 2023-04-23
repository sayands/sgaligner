import os 
import os.path as osp
import argparse
import importlib
import yaml

from utils import common

def parse_config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name(just the file name, not absolute path)')
    args = parser.parse_args()
    return parser, args

def import_cfg(config_file_name):
    directory_path = osp.join(osp.dirname(os.getcwd()), 'configs')
    file_path = osp.join(directory_path, config_file_name)
    spec = importlib.util.spec_from_file_location(config_file_name, file_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    cfg = config.make_cfg()
    return cfg

def load_config(path, make_output_dirs=True):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns: 
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path,'r') as f:
        cfg = yaml.safe_load(f)
    
    config = dict()
    for key, value in cfg.items():
        for k,v in value.items():
            config[k] = v
    
    if make_output_dirs:
        config['working_dir'] = osp.dirname(osp.abspath(__file__))
        config['root_dir']    = osp.dirname(config['working_dir'])
        config.exp_name = '_'.join(config.modules)
        config.output_dir = osp.join(config.root_dir, 'output', config.model_name, config.exp_name)
        config.snapshot_dir = osp.join(config.output_dir, 'snapshots')
        config.log_dir = osp.join(config.output_dir, 'logs')
        config.event_dir = osp.join(config.output_dir, 'events')
    
    return config
