import pickle
import os

def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)