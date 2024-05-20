import numpy as np
import random
from torchvision.datasets import ImageFolder, DatasetFolder

random.seed(1)
np.random.seed(1)


# http://cs231n.stanford.edu/tiny-imagenet-200.zip
# https://github.com/QinbinLi/MOON/blob/6c7a4ed1b1a8c0724fa2976292a667a828e3ff5d/datasets.py#L148
class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        # self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
        '''
        如果self.dataidxs不为None，则代码将使用numpy数组的索引操作来选择子集样本。
        具体来说，它从imagefolder_obj.samples中选择了一部分样本，并将其存储在self.samples中。
        如果self.dataidxs为None，则直接将所有样本存储在self.samples中
        '''

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)