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

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='Depth-aware Transformer for Monocular 3D Object Detection')
parser.add_argument('config', help='settings of detection in yaml format')
parser.add_argument('ckpt', type=str)
args = parser.parse_args()


def main():
    assert (os.path.exists(args.config))
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    set_random_seed(cfg.get('random_seed', 444))
    ctime = datetime.datetime.now()

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % ctime.strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    _, test_loader = build_dataloader(cfg['dataset'])

    # build model
    model, _ = build_model(cfg['model'])
    model = model.cuda()

    output_dir = os.path.join('./' + cfg["trainer"]['save_path'], model_name, ctime.strftime('%Y%m%d_%H%M%S'))
    logger.info('Evaluation Only')
    tester = Tester(cfg=cfg['tester'],
                    model=model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name,
                    checkpoint_path=args.ckpt,
                    output_dir=output_dir)
    tester.test()


if __name__ == '__main__':
    main()
