import torch.nn as nn
import torch.nn.functional as F


def loss_fn_mse(predictions, targets):
    predictions = F.softmax(predictions, dim=1)
    loss = nn.MSELoss()
    return predictions, loss(predictions, targets)

def loss_fn_ce(predictions, targets):
    loss = nn.CrossEntropyLoss()
    return predictions, loss(predictions, targets)

def model_info(sparse, task, loss_mode, hidden_size):
    if sparse:
        print(f'Sparse Mode, with hidden size {hidden_size}')
    else:
        print(f'Dense Mode, with hidden size {hidden_size}')
    print(f'task: {task}, loss_mode: {loss_mode}')