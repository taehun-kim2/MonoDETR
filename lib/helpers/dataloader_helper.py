import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.custom.custom_dataset import Custom_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, device_num=1):
    # perpare dataset
    if cfg['type'] == 'KITTI':
        train_set = KITTI_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set =  KITTI_Dataset(split=cfg['test_split'], cfg=cfg)
    elif cfg['type'] == 'Custom':
        train_set = Custom_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set =  Custom_Dataset(split=cfg['test_split'], cfg=cfg)    
    
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])

    if device_num > 1:
        train_sampler = DistributedSampler(train_set, shuffle=True)
    else:
        train_sampler = None

    # prepare dataloader
    train_loader = DataLoader(dataset=train_set,
                              batch_size=cfg['batch_size'],
                              num_workers=cfg['num_workers'],
                              worker_init_fn=my_worker_init_fn,
                              shuffle=train_sampler is None,
                              sampler=train_sampler,
                              pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=cfg['batch_size'],
                             num_workers=cfg['num_workers'],
                             worker_init_fn=my_worker_init_fn,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    return train_loader, test_loader
