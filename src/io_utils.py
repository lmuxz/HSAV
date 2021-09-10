import os
import gc
import torch
import pickle
import functools
import numpy as np

from torch import nn
from torch.utils import data
from scipy.special import softmax

def save_pickle(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        model = pickle.load(f)
    return model

def load_dataset(dir_path, task, domain, data_type, period):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_train.npy".format(task, domain, data_type, period))
    train = np.load(file_path)

    file_path = os.path.join(dir_path, "{}_{}_{}_{}_train_label.npy".format(task, domain, data_type, period))
    train_label = np.load(file_path)

    file_path = os.path.join(dir_path, "{}_{}_{}_{}_test.npy".format(task, domain, data_type, period))
    test = np.load(file_path)

    file_path = os.path.join(dir_path, "{}_{}_{}_{}_test_label.npy".format(task, domain, data_type, period))
    test_label = np.load(file_path)

    return train, train_label, test, test_label


def cate_to_embedding(dataset, dir_path=None, task=None, domain=None, period=None, version=None, embedding_matrix=None):
    if embedding_matrix is None:
        file_path = os.path.join(dir_path, "{}_{}_embed_{}_{}".format(task, domain, period, version))
        embedding_matrix = load_pickle(file_path)

    dataset_embedding = []
    for i in range(len(embedding_matrix)):
        dataset_embedding.append(embedding_matrix[i][dataset[:, i].astype(int), :])
    dataset_embedding.append(dataset[:, len(embedding_matrix):])

    return np.hstack(dataset_embedding), embedding_matrix


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def embedding_to_cate(dataset, embedding_matrix=None,
    repeat=5, batch_size=256, num_workers=5, device=torch.device("cuda")):
    """
    Stochastic embedding to categorical features transformation. Sample repeat-times cate features according to softmax of cosine distance.
    How to transfer embedding to categorial is still an open question.
    """
    embedding_dim = 0
    for i in range(len(embedding_matrix)):
        embedding_dim += embedding_matrix[i].shape[-1]
    # embedding_dim = functools.reduce(lambda a, b: a.shape[-1] + b.shape[-1], embedding_matrix)

    data_tensor = data.TensorDataset(
        torch.from_numpy(dataset[:, :embedding_dim]).float()
    )
    dataloader = data.DataLoader(data_tensor, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    res = []
    embedding_matrix = [torch.tensor(embedding).to(device) for embedding in embedding_matrix]
    for (tensor,) in dataloader:
        tensor = tensor.to(device)
        cum_index = 0

        dataset_cate = []
        for i in range(len(embedding_matrix)):
            # pairwise cos similarit between i-th feature emebdding and modalities of embedding matrix
            euc_dist = torch.cdist(tensor[:, cum_index:cum_index+embedding_matrix[i].shape[-1]], embedding_matrix[i])
            print("Cate Feature", i, euc_dist)
            euc_dist_min = torch.min(euc_dist, dim=1)[0]
            euc_dist_max = torch.max(euc_dist, dim=1)[0]
            euc_sim = 1 - (euc_dist - euc_dist_min) / (euc_dist_max - euc_dist_min)
            cate_dist = nn.functional.softmax(euc_sim, dim=1)
            # print("Cate Feature", i, cate_dist)
            # dataset_cate_i = torch.multinomial(cate_dist, num_samples=repeat, replacement=True)
            dataset_cate_i = torch.argmax(cate_dist, dim=1)[:, None]
            dataset_cate.append(dataset_cate_i.transpose(0, 1).reshape(-1, 1))
            cum_index += embedding_matrix[i].shape[-1]
        res.append(torch.cat(dataset_cate, dim=1))
    
    return np.hstack([torch.cat(res, dim=0).cpu().numpy(), 
        np.tile(dataset[:, embedding_dim:], (repeat, 1))])


def save_model(model, dir_path, task, domain, model_type, period, version="v0"):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_{}".format(task, domain, model_type, period, version))
    save_pickle(model, file_path)
    print("Model Saved", flush=True)


def load_model(dir_path, task, domain, model_type, period, version="v0"):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_{}".format(task, domain, model_type, period, version))
    model = load_pickle(file_path)

    return model


def model_log(dir_path, task, domain, model_type, period, version, text):
    file_path = os.path.join(dir_path, "{}_{}_{}_{}_{}".format(task, domain, model_type, period, version))
    with open(file_path, "a+") as f:
        f.writelines(text + "\n")

def write_log(path, file, text):
    with open(os.path.join(path, file), "a+") as f:
        f.writelines(text + "\n")
