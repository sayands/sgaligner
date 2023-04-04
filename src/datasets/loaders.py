from datasets.scan3r import Scan3RDataset
from utils import torch_util

def get_train_val_data_loader(cfg):
    train_dataset = Scan3RDataset(cfg, split='train')
    train_dataloader = torch_util.build_dataloader(train_dataset, batch_size=cfg.train.batch_size, num_workers=cfg.num_workers, shuffle=True,
                                                   collate_fn=train_dataset.collate_fn, pin_memory=True, drop_last=True)
    val_dataset = Scan3RDataset(cfg, split='val')
    val_dataloader = torch_util.build_dataloader(val_dataset, batch_size=cfg.val.batch_size, num_workers=cfg.num_workers, shuffle=False,
                                                   collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=True)

    return train_dataloader, val_dataloader

def get_val_dataloader(cfg):
    val_dataset = Scan3RDataset(cfg, split='val')
    val_dataloader = torch_util.build_dataloader(val_dataset, batch_size=cfg.val.batch_size, num_workers=cfg.num_workers, shuffle=False,
                                                   collate_fn=val_dataset.collate_fn, pin_memory=True, drop_last=True)
    return val_dataset, val_dataloader
    
