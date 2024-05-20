from collections import defaultdict
import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.trainmodel.resnet import *
from flcore.trainmodel.models import *

class clientProto(Client):
    def __init__(self, args, id, model_heter, stride, train_samples, test_samples, **kwargs):
        super().__init__(args, id, model_heter, stride,train_samples, test_samples, **kwargs)

        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda

        # 初始化损失和样本量
        self.total_train_num = 1
        self.total_losses = 0

    def train(self):
        start_time = time.time()

        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        total_losses = 0
        for step in range(max_local_epochs):
            total_train_num = 0
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(self.global_protos[y_c]) != type([]):
                            proto_new[i, :] = self.global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_num += y.shape[0]
                total_losses += loss.item() * y.shape[0]

        self.total_train_num = total_train_num
        self.total_losses = total_losses / self.local_epochs

        # self.model.cpu()
        # rep = self.model.base(x)
        # print(torch.sum(rep!=0).item() / rep.numel())

        self.collect_protos()
        # self.protos = agg_func(protos)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

    def collect_protos(self):
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloaderfull):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)

    def test_metrics(self, model=None):
        if model == None:
            model = self.model
        model.eval()

        test_correct_l, test_correct_g = 0, 0
        test_num = 0
        with torch.no_grad():
            for x, y in self.testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output_l = self.model.head(rep)

                test_correct_l += (torch.sum(torch.argmax(output_l, dim=1) == y)).item()
                test_num += y.shape[0]

                if self.global_protos is not None:
                    output_g = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in self.global_protos.items():
                            if type(pro) != type([]):
                                output_g[i, j] = self.loss_mse(r, pro)
                    test_correct_g += (torch.sum(torch.argmin(output_g, dim=1) == y)).item()
                else:
                    test_correct_g = 0

        return test_correct_l, test_correct_g, test_num

    def train_metrics(self):
        return self.total_losses, self.total_train_num

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:# 把同一个label的proto全局加起来
                proto += i.data
            protos[label] = proto / len(proto_list)# 再除以该拥有的proto数量
        else:
            protos[label] = proto_list[0]

    return protos