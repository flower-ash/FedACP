import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from flcore.trainmodel.models import *
from flcore.trainmodel.resnet import *

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, model_heter, stride,train_samples, test_samples, **kwargs):
        self.num_clients = args.num_clients
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.alpha = args.alpha #异构程度

        self.partition = args.partition
        self.alpha = args.alpha
        self.n = args.n
        self.k = args.k

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        # 如果模型异构，则重新构建本地模型
        if model_heter:
            if self.dataset == "Cifar10":
                self.model = resnet18(stride, pretrained=False, num_classes=args.num_classes).to(args.device)
            elif self.dataset == "Cifar100" or self.dataset == "Tiny_imagenet":
                self.model = resnet50(stride, pretrained=False, num_classes=args.num_classes).to(args.device)
            head = copy.deepcopy(self.model.fc)
            self.model.fc = nn.Identity()
            self.model = BaseHeadSplit(self.model, head)

        else:
            self.model = copy.deepcopy(args.model)  # 客户端本地模型

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.loss = nn.CrossEntropyLoss().to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

        self.testloaderfull = self.load_test_data()
        self.trainloader = self.load_train_data(drop=True)
        self.trainloaderfull = self.load_train_data(drop=False)

    def load_train_data(self, batch_size=None, drop = True):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.num_clients, self.partition, self.alpha, self.id, self.n, self.k, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=drop, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.num_clients,self.partition, self.alpha, self.id, self.n, self.k, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        self.model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        with torch.no_grad():
            for x, y in self.testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        # average='micro': 这是一个可选参数，用于指定如何计算多分类问题的AUC。在二分类问题中，AUC的计算很简单，只有一个值。但在多分类问题中，有多个类别，因此需要指定如何合并这些类别的AUC。
        # 在本例中，指定average='micro'表示采用微平均（micro-average）的方法来计算多分类AUC。微平均将所有样本的真阳性、假阳性、真阴性、假阴性的数量汇总，然后计算合并后的AUC。
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num



    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
