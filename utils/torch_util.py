import os 
import os.path as osp
import torch

def save_model(state, file_name, logger=None):
    torch.save(state, file_name)
    if logger: logger.info("Write snapshot into {}".format(file_name))

def load_model(model, file_name, logger=None):
    ckpt = torch.load(file_name)
    
    model.load_state_dict(ckpt['network'])
    if logger: logger.info("Loaded snapshot from {}".format(file_name))
    return model
    