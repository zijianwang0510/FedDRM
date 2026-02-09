import sys, os

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)

import argparse
from omegaconf import OmegaConf
from utils.utils import get_log_path, create_path, seed_everything, log_everything, select_algorithm, load_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='fedavg, feddrm, fedrep, etc...')

    args = parser.parse_args()

    config_path = os.path.join(base_path, 'configs', f'{args.config}.yaml')
    cfg = OmegaConf.load(config_path)

    create_path(cfg)

    with open(os.path.join(get_log_path(cfg), 'run.log'), 'a') as logfile:
        log_everything(cfg, logfile)

        seed_everything(cfg.seed)

        train_loaders, test_loaders, num_data = load_data(cfg)

        FedAlgo = select_algorithm(cfg)

        algo = FedAlgo(cfg, train_loaders, test_loaders, num_data, logfile)

        algo.train()