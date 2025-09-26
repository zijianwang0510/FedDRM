import sys, os

base_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_path)

import random
import numpy as np
import torch
from datetime import datetime

def seed_everything(seed):
    # Set Python's built-in random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # If using CUDA, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Disable PyTorch's deterministic algorithms if needed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to {seed}...")


def save_path(cfg):
    if 'feddrm' in cfg.algo:
        if cfg.ablation_arch == True:
            ret = f"{cfg.algo}_{cfg.feddrm_ratio}_arch_{cfg.share_level}_{cfg.dataset}_{cfg.partition}_{cfg.dir_alpha if cfg.partition == 'dir' else cfg.num_shards}_T_{cfg.commu_round}_E_{cfg.local_steps}_{cfg.model}"
        elif cfg.ablation_feature_shift == True:
            ret = f"{cfg.algo}_{cfg.feddrm_ratio}_shift_{cfg.shift_level}_{cfg.dataset}_{cfg.partition}_{cfg.dir_alpha if cfg.partition == 'dir' else cfg.num_shards}_T_{cfg.commu_round}_E_{cfg.local_steps}_{cfg.model}"
        else:
            ret = f"{cfg.algo}_{cfg.feddrm_ratio}_{cfg.dataset}_{cfg.partition}_{cfg.dir_alpha if cfg.partition == 'dir' else cfg.num_shards}_T_{cfg.commu_round}_E_{cfg.local_steps}_{cfg.model}"
    else:
        ret = f"{cfg.algo}_{cfg.dataset}_{cfg.partition}_{cfg.dir_alpha if cfg.partition == 'dir' else cfg.num_shards}_T_{cfg.commu_round}_E_{cfg.local_steps}_{cfg.model}"
        if 'ft' in cfg.algo:
            ret += f"_FT_{cfg.ft_round}"
    
    if cfg.sensitivity_num_clients == True:
        ret += f"_N_{cfg.num_clients}"

    return ret


def create_path(cfg):
    os.makedirs(os.path.join(base_path, 'logs', save_path(cfg)), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'save', save_path(cfg)), exist_ok=True)


def get_log_path(cfg):
    return os.path.join(base_path, 'logs', save_path(cfg))


def get_local_model_path(cfg, idx):
    return os.path.join(base_path, 'save', save_path(cfg), f'local_model_{idx}.pth')


def get_global_model_path(cfg):
    return os.path.join(base_path, 'save', save_path(cfg), 'global_model.pth')


# Added for Ditto
def get_personalized_model_path(cfg, idx):
    return os.path.join(base_path, 'save', save_path(cfg), f'personalized_model_{idx}.pth')


# Added for FedAS
def get_prev_local_model_path(cfg, idx):
    return os.path.join(base_path, 'save', save_path(cfg), f'prev_local_model_{idx}.pth')


def get_record_path(cfg):
    return os.path.join(base_path, 'save', save_path(cfg), 'record.npy')


def log_everything(cfg, logfile):
    logfile.write(f'==={datetime.now():%Y-%m-%d %H:%M:%S}===\n')
    logfile.write('===Setting===\n')

    for k, v in cfg.items():
        logfile.write(f'{k}: {v}\n')

    logfile.write('===End of Setting===\n\n')
    logfile.flush()


def select_model(cfg):
    from models.nets import LeNetGN, ResNet
    if cfg.model == 'lenetgn':
        model = LeNetGN
    elif cfg.model == 'resnet':
        model = ResNet
    else:
        raise ValueError(f"Model {cfg.model} is not supported...")

    print(f"Model selected: {cfg.model}")

    return model(cfg).to(cfg.device)


def select_algorithm(cfg):
    from algorithms.feddrm import FedDRM
    from algorithms.fedrep import FedRep
    from algorithms.fedavgft import FedAvgFT
    from algorithms.fedproxft import FedProxFT
    from algorithms.fedsamft import FedSAMFT
    from algorithms.ditto import Ditto
    from algorithms.fedbabu import FedBABU
    from algorithms.fedas import FedAS
    from algorithms.fedpac import FedPAC
    from algorithms.fedala import FedALA
    from algorithms.confree import ConFREE

    if cfg.algo == 'feddrm': 
        FedAlgo = FedDRM
    elif cfg.algo == 'fedrep':
        FedAlgo = FedRep
    elif cfg.algo == 'fedavgft':
        FedAlgo = FedAvgFT
    elif cfg.algo == 'fedproxft':
        FedAlgo = FedProxFT
    elif cfg.algo == 'ditto':
        FedAlgo = Ditto
    elif cfg.algo == 'fedsamft':
        FedAlgo = FedSAMFT
    elif cfg.algo == 'fedbabu':
        FedAlgo = FedBABU
    elif cfg.algo == 'fedas':
        FedAlgo = FedAS
    elif cfg.algo == 'fedpac':
        FedAlgo = FedPAC
    elif cfg.algo == 'confree':
        FedAlgo = ConFREE
    elif cfg.algo == 'fedala':
        FedAlgo = FedALA
    else:
        raise ValueError(f'Algorithm {cfg.algo} is not supported...')
    
    return FedAlgo