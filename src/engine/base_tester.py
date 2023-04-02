import os 
import os.path as osp

import torch 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import common
from utils.logger import colorlogger
from aligner.sg_aligner import *
from aligner.losses import *

from datasets import get_dataset
from base import Base
from utils import torch_util

class BaseTester(Base):
    def __init__(self, cfg):
        super(BaseTester, self).__init__() 
        self.modules = cfg.val.modules
        self.module_name = '_'.join(self.modules)
        self.dataset_name = cfg.data.name
        
        # setup logging directories
        self.out_dir = osp.join(os.getcwd(), 'out_{}/out_{}'.format(self.dataset_name, self.module_name))
        self.log_dir = osp.join(self.out_dir, 'logs')
        self.model_dir = osp.join(self.out_dir, 'models')

        common.assert_dir(self.model_dir)

        # logger        
        self.logger = colorlogger(self.log_dir, log_name='log.txt')

        # tensorboard writer
        self.writer = SummaryWriter(self.log_dir)

        self.num_workers = cfg.num_workers
        self.batch_size = cfg.val.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.split = 'val'

        self.register_model()
        self._make_generator()

    def load_snapshot(self, snapshot):
        torch_util.load_model(self.model, snapshot)

    def register_model(self):
        self.logger.info("Creating graph and optimizer...")
        self.model = MultiModalEncoder(train_modules=self.modules, rel_dim=self.cfg.model.rel_dim, attr_dim=self.cfg.model.attr_dim)
        self.model.to(self.device)
        self.model.eval()
    
    def _make_generator(self):
        self.dataset = get_dataset(self.dataset_name)
        self.logger.info("Creating {} dataset of {} samples...".format(self.split, len(self.dataset)))
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn = self.dataset.collate_fn,
                                      num_workers=self.num_workers, pin_memory=True, drop_last=True)
        self.iter_per_epoch = self.data_loader.__len__()
        
    
