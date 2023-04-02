import os 
import os.path as osp
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
import torch.backends.cudnn as cudnn
import random

def save_model(state, file_name, logger=None):
    torch.save(state, file_name)
    if logger: logger.info("Write snapshot into {}".format(file_name))

def load_model(model, file_name, logger=None):
    ckpt = torch.load(file_name)
    
    model.load_state_dict(ckpt['network'])
    if logger: logger.info("Loaded snapshot from {}".format(file_name))
    return model

def release_cuda(x):
    r"""Release all tensors to item or numpy array."""
    if isinstance(x, list):
        x = [release_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (release_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: release_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.detach().cpu().numpy()
    return x

def to_cuda(x):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x

def initialize(seed=None, cudnn_deterministic=True, autograd_anomaly_detection=False):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
    if cudnn_deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        cudnn.benchmark = True
        cudnn.deterministic = False
    torch.autograd.set_detect_anomaly(autograd_anomaly_detection)

def all_reduce_tensor(tensor, world_size=1):
    r"""Average reduce a tensor across all workers."""
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor)
    reduced_tensor /= world_size
    return reduced_tensor


def all_reduce_tensors(x, world_size=1):
    r"""Average reduce all tensors across all workers."""
    if isinstance(x, list):
        x = [all_reduce_tensors(item, world_size=world_size) for item in x]
    elif isinstance(x, tuple):
        x = (all_reduce_tensors(item, world_size=world_size) for item in x)
    elif isinstance(x, dict):
        x = {key: all_reduce_tensors(value, world_size=world_size) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = all_reduce_tensor(x, world_size=world_size)
    return x

def reset_seed_worker_init_fn(worker_id):
    r"""Reset seed for data loader worker."""
    seed = torch.initial_seed() % (2 ** 32)
    # print(worker_id, seed)
    np.random.seed(seed)
    random.seed(seed)

def build_dataloader(
    dataset,
    batch_size=1,
    num_workers=1,
    shuffle=None,
    collate_fn=None,
    pin_memory=False,
    drop_last=False,
    distributed=False,
):
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = shuffle

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_fn,
        worker_init_fn=reset_seed_worker_init_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return data_loader