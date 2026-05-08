import torch
import torch.nn as nn
from models.Classifier.activation import getActivation


class FFNClassifier(nn.Module):
    def __init__(self, in_ft, out_ft_list, activation, drop_prob, n_cls):
        super(FFNClassifier, self).__init__()

        self.proj = nn.Linear(in_ft, out_ft_list[0])
        self.linear1 = nn.Linear(out_ft_list[0], out_ft_list[1])
        self.classifier = nn.Linear(out_ft_list[1], n_cls)

        self.act = getActivation(activation)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.act(self.proj(x))
        x = self.drop(x)
        
        x = self.act(self.linear1(x))
        x = self.drop(x)
        
        x = self.classifier(x)

        return x



def TrainFFN(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    if len(loader) == 0:
        return 0

    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def EvaluateFFN(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    if len(loader) == 0:
        return 0

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()
    
    return total_loss / len(loader)