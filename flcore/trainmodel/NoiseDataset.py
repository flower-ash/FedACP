import torch
from torch.utils.data import Dataset

class NoiseDataset(Dataset):
    def __init__(self, data):
        """
            Args:
                data (list): 包含输入数据的列表
        """
        self.data = data

    def __len__(self):
        """
        返回数据集的总样本数
        """
        return len(self.data)

    def __getitem__(self, index):
        """
            根据给定索引返回一个样本
        """
        x = self.data[index]
        return x