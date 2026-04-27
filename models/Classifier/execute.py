import os

import torch
import torch.nn as nn

from models.DGI.models.dgi import DGI
from models.DGI.execute import extract_embeddings as DGI_extract_embeddings

from models.Classifier.FFN import TrainFFN


def run_downstream(config, clf_model, train_loader, val_loader, test_loader):

    MODEL_PATH = os.path.join(config.SAVEPATH, config.load_model, config.load_wandb_id, 'BestPerformance.pth')
    if os.path.exists(MODEL_PATH):
        pretrained_model.load_state_dict(torch.load(MODEL_PATH))


    # print("\n" + "="*60)
    # print("EXTRACTING EMBEDDINGS")
    # print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if config.load_model == 'DGI':
        train_embeds, _ = DGI_extract_embeddings(pretrained_model, train_loader, device)
        val_embeds, _ = DGI_extract_embeddings(pretrained_model, val_loader, device)
        test_embeds, _ = DGI_extract_embeddings(pretrained_model, test_loader, device)

    elif config.load_model == 'XXXModel':
        pass


    # Downstream Task
    if config.clf_model == 'FFN':
        clf_model.to(device)



        
        # Train FFN

        # Evaluate FFN

        # Save Results


    elif 

    