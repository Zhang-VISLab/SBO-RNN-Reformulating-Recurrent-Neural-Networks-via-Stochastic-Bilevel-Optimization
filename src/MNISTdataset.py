import torchvision
import numpy as np
from utils.dataset_utils import dataset_info, norm_mean_var, norm_single_mean_var, generate_permutation_seq, scalar2vector
import torch as torch
import random
from torch.utils.data import DataLoader

class MNIST():
    def __init__(self, dataset, ratio, istrain, norm_mode, loss_mode, nfeatures=28):
        self.data = torchvision.datasets.MNIST(
            root='../data',
            train=istrain,
            download=True
        )
        self.norm_mode = norm_mode
        self.loss_mode = loss_mode
        self.nfeatures = nfeatures
        self.len = int(ratio * len(self.data))
        self.dataset = dataset
        self.ids = random.sample([i for i in range(len(self.data))], self.len)
        dataset_info(dataset, self.len, istrain, norm_mode, nfeatures)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        img, target = self.data[self.ids[item]]
        img = np.array(img)

        if self.norm_mode == 1:
            img = norm_mean_var(img)
        elif self.norm_mode == 2:
            img = norm_single_mean_var(img)
        else:
            pass

        if self.dataset == 'pixel':
            pass
        elif self.dataset == 'permute':
            perm_seq = generate_permutation_seq()
            img = img.reshape(1, 784)
            img = img[:, perm_seq]

        img = img.reshape(-1, self.nfeatures)


        if self.loss_mode == 'ce':
            return{
                'img': torch.tensor(img, dtype=torch.float),
                'target': torch.tensor(target, dtype=torch.long)
            }
        elif self.loss_mode == 'mse':
            target = scalar2vector(target)
            return {
                'img': torch.tensor(img, dtype=torch.float),
                'target': torch.tensor(target, dtype=torch.float)
            }
if __name__ == '__main__':
    testset = MNIST(dataset='pixel', ratio=0.01, istrain=False, norm_mode=1, loss_mode='mse', nfeatures=28)
    dataloader = DataLoader(testset, batch_size=2, shuffle=False)
    # print(len(dataloader))
    for batch_idx, dic in enumerate(dataloader):
        print('img', dic['img'].shape)
        print('target', dic['target'].shape)







