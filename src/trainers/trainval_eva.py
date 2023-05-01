import argparse
import time

import torch.optim as optim

import sys
sys.path.append('.')

from engine import EpochBasedTrainer
from datasets.loaders import get_train_val_data_loader
from aligner.eva import *
from aligner.losses import OverallNCALoss

from configs import update_config, config

class EVATrainer(EpochBasedTrainer):
    def __init__(self, cfg,parser=None):
        super().__init__(cfg, parser)
        
        # Model Specific params
        self.modules = cfg.modules
        self.rel_dim = cfg.model.rel_dim
        self.attr_dim = cfg.model.attr_dim
        
        # Loss params
        
        # dataloader
        start_time = time.time()
        train_loader, val_loader = get_train_val_data_loader(cfg)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model
        model = self.create_model()
        self.register_model(model)

        # loss function and params
        self.loss_func = OverallNCALoss(modules = self.modules, device=self.device)
        self.params = [{'params' : list(self.model.parameters())}]
        
        # optimizer and scheduler
        optimizer = optim.Adam(self.params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        self.logger.info('Initialisation Complete')

    def create_model(self):
        model = EVA(modules=self.modules, rel_dim = self.rel_dim, attr_dim=self.attr_dim).to(self.device)
        message = 'Model created'
        self.logger.info(message)
        return model

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        return output_dict, loss_dict

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

def parse_args(parser=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', default='', type=str, help='configuration file name')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--epoch', type=int, default=None, help='load epoch')
    parser.add_argument('--log_steps', type=int, default=500, help='logging steps')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for ddp')

    args = parser.parse_args()
    return parser, args
    
def main():
    parser, args = parse_args()
    cfg = update_config(config, args.config)
    trainer = EVATrainer(cfg, parser)
    trainer.run()

if __name__ == '__main__':
    main()









