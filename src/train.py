import torch
import pandas as pd
import optuna
from tqdm import tqdm
import joblib
from easydict import EasyDict
import sys
sys.path.append('/workspace/')
from utils.base_utils import ensure_dir, save_pickle, read_pickle
from utils.train_utils import target_format, get_acc, setup_seed

from MNISTdataset import MNIST
from models import ODE_Vanilla

import warnings
warnings.filterwarnings('ignore')

DEVICE = 'cuda'

def trian_loop_fn(dataset, dataloader, model, optimizer, loss_mode):
    model.train()
    loss_sum = 0
    counter = 0

    for step, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
        counter += 1

        img = data['img']
        target = data['target']
        img = img.to(DEVICE, dtype=torch.float)
        target = target_format(target, loss_mode, device=DEVICE)

        optimizer.zero_grad()
        _, loss = model(img, target)

        loss_sum += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()
    return loss_sum / counter

def eval_loop_fn(dataset, dataloader, model, loss_mode):
    model.eval()
    loss_sum = 0
    counter = 0
    correct = 0
    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1

            img = data['img']
            target = data['target']
            img = img.to(DEVICE, dtype=torch.float)
            target = target_format(target, loss_mode, device=DEVICE)

            x, loss = model(img, target)
            loss_sum += loss.item()

            correct += get_acc(x, target, loss_mode)
    return loss_sum / counter, 100. * correct / len(dataset)

def run(params, save_model=False, path=''):
    model = ODE_Vanilla(
        sparse=params['sparse'], task=params['task'], loss_mode=params['loss_mode'],
        hidden_size=params['hidden_size'], nfeatures=params['nfeatures'], device='cuda',
        eta=params['eta'], mu=params['mu']
    )
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    trian_dataset = MNIST(
        dataset=params['dataset'], ratio=params['train_ratio'], istrain=True,
        norm_mode=params['norm_mode'], loss_mode=params['loss_mode'], nfeatures=params['nfeatures']
    )
    trian_dataloader = torch.utils.data.DataLoader(
        dataset=trian_dataset,
        batch_size=params['train_batch_size'],
        shuffle=True,
        num_workers=1
    )

    valid_dataset = MNIST(
        dataset=params['dataset'], ratio=params['valid_ratio'], istrain=False,
        norm_mode=params['norm_mode'], loss_mode=params['loss_mode'], nfeatures=params['nfeatures']
    )
    valid_dataloader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=params['valid_batch_size'],
        shuffle=False,
        num_workers=1
    )

    dfhistory = pd.DataFrame(columns=['epoch', 'train_loss', 'valid_loss', 'valid_acc'])

    best_acc = -1
    best_acc_epoch = -1
    for epoch in range(1, params['epoch'] + 1):
        train_loss = trian_loop_fn(trian_dataset, trian_dataloader, model, optimizer, loss_mode=params['loss_mode'])
        valid_loss, valid_acc = eval_loop_fn(valid_dataset, valid_dataloader, model, loss_mode=params['loss_mode'])


        if valid_acc > best_acc:
            best_acc = round(valid_acc, 4)
            best_acc_epoch = epoch
            if save_model:
                torch.save(model.state_dict(), f'{path}ODE_Vanilla_acc@epoch{epoch}.bin')

        print(f'epoch: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}, valid_acc: {valid_acc}')
        print(f'best_acc: {best_acc} @ epoch {best_acc_epoch}')
        scheduler.step()

        if save_model:
            info = (int(epoch), train_loss, valid_loss, valid_acc)
            dfhistory.loc[epoch - 1] = info
            dfhistory.to_csv(f'{path}dfhistory.csv', index=False)
            if epoch % 10 == 0:
                torch.save(model.state_dict(), f'{path}ODE_Vanilla_@epoch{epoch}.bin')
    if save_model:
        torch.save(model.state_dict(), f'{path}ODE_Vanilla_@epoch{params["epoch"]}.bin')
    return best_acc



if __name__ == '__main__':
    setup_seed(2021)

    params = EasyDict(
        dataset='pixel',  # permute, pixel
        sparse=False, # sparse or dense
        task='GD',  # GD, HB, NAG
        loss_mode='ce',  # mse for har-2, ce for MNIST
        train_ratio=1, # the ratio will be applied for training set
        valid_ratio=1, # the ratio will be applied for test set (suggest using 1)
        lr=1e-2, # learning rate
        eta=1e-3, # eta
        mu=0.5, # mu
        norm_mode=1,  # 1 for dataset mean variance normalizaiton, 2 for data mean variance normalization
        nfeatures=1,  # 28 for har-2, 1 for MINIST
        hidden_size=128, # hidden size for recurrent block
        epoch=200,
        train_batch_size=32,
        valid_batch_size=32,
    )

    path = f'../{params["dataset"]}/{params["tasks"]}/{params["lr"]}_{params["eta"]}_{params["mu"]}/'
    ensure_dir(path)

    run(params, save_model=True, path=path)
