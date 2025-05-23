import numpy as np
import os
import torch


def read_data(dataset, num_clients, partition, alpha, idx, n, k, is_train=True):
    if partition == "dir":
        data_dir = os.path.join('/root/autodl-tmp/FedCPL/data', dataset + "_dir" + "_nc" + str(num_clients) + "_a" + str(alpha))
    elif partition == "pat":
        data_dir = os.path.join('/root/autodl-tmp/FedCPL/data',dataset + "_pat" + "_n" + str(n) + "_k" + str(k))

    if is_train:
        train_data_dir = os.path.join(data_dir, 'train/')
        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(data_dir, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, num_clients, partition, alpha, idx, n, k, is_train=True):
    if is_train:
        train_data = read_data(dataset,num_clients, partition, alpha, idx, n, k, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, num_clients, partition, alpha, idx, n, k, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


