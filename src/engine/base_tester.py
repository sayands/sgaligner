import sys
import argparse
import os.path as osp
import time
import json
import abc

import torch

from utils import torch_util
from utils.logger import Logger
from utils.timer import Timer

class BaseTester(abc.ABC):
    def __init__(self, cfg, parser=None, cudnn_deterministic=True):
        # parser
        self.args = parser.parse_args()

        # logger
        log_file = osp.join(cfg.log_dir, 'test-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
        self.logger = Logger(log_file=log_file)

        # command executed
        message = 'Command executed: ' + ' '.join(sys.argv)
        self.logger.info(message)

        # find snapshot
        if self.args.snapshot is None:
            if self.args.test_epoch is not None:
                self.args.snapshot = osp.join(cfg.snapshot_dir, 'epoch-{}.pth.tar'.format(self.args.test_epoch))
            elif self.args.test_iter is not None:
                self.args.snapshot = osp.join(cfg.snapshot_dir, 'iter-{}.pth.tar'.format(self.args.test_iter))
        if self.args.snapshot is None:
            raise RuntimeError('Snapshot is not specified.')

        # print config
        message = 'Configs:\n' + json.dumps(cfg, indent=4)
        self.logger.info(message)

        # cuda and distributed
        if not torch.cuda.is_available(): raise RuntimeError('No CUDA devices available.')
        self.device = torch.device("cuda")
        
        self.cudnn_deterministic = cudnn_deterministic
        self.seed = cfg.seed
        torch_util.initialize(seed=self.seed, cudnn_deterministic=self.cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        self.test_loader = None
        self.saved_states = {}

        self.timer = Timer()

    def load_snapshot(self, snapshot):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=self.device)
        assert 'model' in state_dict, 'No model can be loaded.'
        self.model.load_state_dict(state_dict['model'], strict=True)
        self.logger.info('Model has been loaded.')

    def register_model(self, model):
        r"""Register model. DDP is automatically used."""
        self.model = model
        message = 'Model description:\n' + str(model)
        self.logger.info(message)
        return model
    def register_dataset(self, test_dataset):
        r"""Register data set."""
        self.test_dataset = test_dataset
    def register_loader(self, test_loader):
        r"""Register data loader."""
        self.test_loader = test_loader

    @abc.abstractmethod
    def run(self):
        raise NotImplemented