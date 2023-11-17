import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

import torch.nn as nn
import torch.cuda as cuda
import torch.distributed as dist

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('config', help='settings of detection in yaml format')
args = parser.parse_args()

def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))
    ctime = datetime.datetime.now()
    
    cuda_visible_devices = None
    local_rank = -1

    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        cuda_visible_devices = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
    if "LOCAL_RANK" in os.environ.keys():
        local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank == -1:
        device_num = 1
    elif cuda_visible_devices is None:
        device_num = torch.cuda.device_count()
    else:
        device_num = len(cuda_visible_devices)

    cfg['trainer']['device_num'] = device_num
    cfg['trainer']['local_rank'] = local_rank

    if device_num > 1:
        cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.device_num, timeout=datetime.timedelta(seconds=3600))
        
    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    if local_rank <= 0:
        os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % ctime.strftime('%Y%m%d_%H%M%S'))
    if local_rank <= 0:
        logger = create_logger(log_file)
    else:
        logger = None

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'], device_num=device_num)
        
    # build model
    model, loss = build_model(cfg['model'])
    
    if device_num > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    else:
        model = model.cuda()
        
    #ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    output_dir = os.path.join('./' + cfg["trainer"]['save_path'], model_name, ctime.strftime('%Y%m%d_%H%M%S'))

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name,
                      output_dir=output_dir)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name,
                    output_dir=output_dir)
    
    if local_rank <= 0:
        trainer.tester = tester

    if logger is not None:
        logger.info('Training')
        logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
        logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    if logger is not None:
        logger.info('Evaluation')
        logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
        logger.info('Split: %s' % (cfg['dataset']['test_split']))

    if local_rank <= 0:
        tester.test()


if __name__ == '__main__':
    main()
