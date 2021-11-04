import numpy as np

def norm_single_mean_var(img):
    mean = np.mean(img)
    std = np.mean(img)
    return (img - mean) / std


def norm_mean_var(img):
    return (img - 0.1307) / 0.3081


def generate_permutation_seq(seed=12008):
    rng = np.random.RandomState(seed)
    return rng.permutation(784)

def scalar2vector(scalar, r=10):
    v = np.zeros((r))
    v[scalar] = 1
    return v

def dataset_info(dataset, length, istrain, norm_mode, nfeatures):
    print(f'MNIST {dataset}')
    if istrain:
        print(f'Train , length: {length}')
    else:
        print(f'Valid , length: {length}')
    if norm_mode == 1:
        print(f'norm_mean_var, with {nfeatures} features')
    elif norm_mode == 2:
        print(f'norm_single_mean_var, with {nfeatures} features')
    else:
        print(f'no normalization, with {nfeatures} features')