import os
import yaml
import argparse

import gc
import torch
import random
import numpy as np
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

def init_wandb(config):
    import wandb
    if config.nowandb:
        print("Do Not Use Wandb")
        return None
    
    if config.wandb_key is not None:
        wandb.login(key=config.wandb_key)
    else:
        if config.nowandb:
            return None
        else:
            raise ValueError("wandb_key is not provided")
    
    init_args = {
        "project": config.project_name,
        "entity": config.entity_name,
        "name": config.wandb_run_name,
    }

    if config.load_pretrained:
        assert config.wandb_run_id is not None, "wandb_run_id is not provided"
        run = wandb.init(
            **init_args,
            id=config.wandb_run_id,
            resume='must',
            reinit=True,
            settings=wandb.Settings(init_timeout=60)
        )

    else:
        run = wandb.init(
            **init_args,
            config=config,
            settings=wandb.Settings(init_timeout=60)
            )
    
    return run

def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def formatTime(training_time):
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    formatted_time = f"{hours}:{minutes:02}:{seconds:02}"

    return formatted_time

def today_date():
    """
    Returns today's date in MMDDYY format.
    Example: 040126
    """
    return datetime.now().strftime('%m%d%y')

def clean_the_memory():
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_optimized_n_jobs(n_jobs_input):
    """
    Check if running under Slurm and return the allocated CPU count.
    If not in Slurm, return the user-specified n_jobs or CPU count.
    """
    # Check if 'SLURM_CPUS_ON_NODE' exists (Slurm allocation variable)
    slurm_cpus = os.getenv('SLURM_CPUS_ON_NODE')
    
    if slurm_cpus:
        # Use Slurm's exact allocation
        allocated_cpus = int(slurm_cpus)
        print(f"Slurm detected: Using allocated {allocated_cpus} CPUs.")
        return allocated_cpus
    else:
        # Fallback to local CPU count if not in Slurm
        print("Slurm not detected: Using default system CPUs.")
        return multiprocessing.cpu_count() if n_jobs_input == -1 else n_jobs_input - 1