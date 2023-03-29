import os 
import os.path as osp

import torch 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils import common, torch_util
from utils.logger import colorlogger
from utils.timer import Timer
from aligner.sg_aligner import *
from aligner.losses import *

from datasets.scan3r import Scan3RDataset
from base import Base

class BaseTrainer(Base):
    def __init__(self, cfg):
        super(BaseTrainer, self).__init__()

        self.do_augmentation = cfg.train.use_augmentation
 
        self.modules = cfg.train.modules

        self.module_name = '_'.join(self.modules)
        self.dataset_name = cfg.data.name
        
        # setup logging directories
        self.out_dir = osp.join(os.getcwd(), 'out_{}/out_{}'.format(self.dataset_name, self.module_name))
        self.log_dir = osp.join(self.out_dir, 'logs')
        self.model_dir = osp.join(self.out_dir, 'models')
        
        common.ensure_dir(self.out_dir)
        common.ensure_dir(self.log_dir)
        common.ensure_dir(self.model_dir)

        # logger        
        self.logger = colorlogger(self.log_dir, log_name='log.txt')

        # tensorboard writer
        self.writer = SummaryWriter(self.log_dir)

        self.num_workers = cfg.train.num_workers
        self.start_epoch = 0
        self.end_epoch = cfg.train.end_epoch
        self.batch_size = cfg.train.batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.split = 'train'

        self._make_model()
        self._make_generator()

    def _make_model(self):
        self.logger.info("Creating graph and optimizer...")
        self.model = MultiModalEncoder(train_modules=self.modules, rel_dim=self.cfg.model.rel_dim, attr_dim=self.cfg.model.attr_dim)
        self.model.to(self.device)
        
        if len(self.modules) > 1:
            self.loss_layer_icl = CustomMultiLossLayer(loss_num=len(self.modules)).to(self.device)
            self.loss_layer_ial = CustomMultiLossLayer(loss_num=len(self.modules)).to(self.device)
            self.params = [{ 'params' : list(self.model.parameters()) + list(self.loss_layer_icl.parameters()) + list(self.loss_layer_ial.parameters())}]
        else:
            self.params = [{'params' : list(self.model.parameters())}]
        
        self.lr = self.cfg.train.learning_rate
        self.optimizer = optim.AdamW(self.params, lr=self.lr)
        self.criterionCl = ICLLoss(device=self.device)
        self.criterionAlign = IALLoss(device=self.device)
    
    def _make_generator(self):
        self.dataset = Scan3RDataset(self.cfg, self.split)
        self.logger.info("Creating {} dataset of {} samples...".format(self.split, len(self.dataset)))
        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, collate_fn = self.dataset.collate_fn,
                                        num_workers=self.num_workers, pin_memory=True, drop_last=True)
        self.iter_per_epoch = self.data_loader.__len__()
        
    
