import os
import yaml
import argparse

import random
import numpy as np

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

def LoadConfig(args):
    with open(args.config_path, 'r') as f:
        run_config = yaml.safe_load(f)
    
    model_name = run_config.get('model')
    model_config_path = os.path.join(
        os.path.dirname(__file__), f'../config/{model_name}.yaml'
    )
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    combined_config = {**model_config, **run_config, **vars(args)}

    for key, value in combined_config.items():
        if value == "None":
            combined_config[key] = None

    return argparse.Namespace(**combined_config)

def print_time(training_time):
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    formatted_time = f"{hours}:{minutes:02}:{seconds:02}"

    return formatted_time

def clean_the_memory():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed):
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)