from collections import defaultdict
import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from flcore.trainmodel.resnet import *
from flcore.trainmodel.models import *

class ClientCP(Client):
    def __init__(self, args, id, model_heter, stride, train_samples, test_samples, **kwargs):
        super().__init__(args, id, model_heter, stride, train_samples, test_samples, **kwargs)
        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()

        # 重新弄个参数
        self.temp = args.temp
        self.num_classes = args.num_classes
        self.beta = args.beta


        # self.old_model = copy.deepcopy(self.model)
        self.confidence_w = torch.zeros(self.num_classes) #初始化置信度权重

        # 初始化损失和样本量
        self.total_train_num = 1
        self.total_losses = 0

        self.sample_per_class = torch.zeros(self.num_classes)  # 用来存放该客户端每一类的样本数
        for x, y in self.trainloader :  # x是样本，y是标签
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        # tensor.item()：将张量中的单个元素转换为Python标量

        self.is_ws = args.is_ws


    def train(self):
        start_time = time.time()

        self.model.to(self.device)
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
                CE_loss = self.loss(output, y)

                # if self.global_protos is not None:
                #     # proto_postive = copy.deepcopy(rep.detach())
                #     # # proto_negative_update = copy.deepcopy(rep.detach())
                #     # negative_protos_list = []
                #     l_sc = None
                #     for i, yy in enumerate(y):# 遍历每个样本
                #         y_c = yy.item()#查看该样本标签
                #         proto_postive = self.global_protos[y_c].detach()#收集与该样本标签一致的proto作为正样本proto
                #         samp_rep = rep[i, :]
                #         numerator = torch.exp(samp_rep.matmul(proto_postive))
                #
                #         proto_list = [torch.tensor(proto) for proto in self.global_protos.values()]
                #         global_proto = torch.stack(proto_list)
                #
                #         denominator = torch.mm(global_proto, samp_rep.view(-1, 1))
                #         denominator = torch.sum(torch.exp(denominator))
                #         if l_sc == None:
                #             l_sc = -1 * torch.log(numerator / denominator)
                #         else:
                #             l_sc += -1 * torch.log(numerator / denominator)
                #
                #     loss_cp = l_sc / self.batch_size
                #     loss = self.beta * loss_cp + (1-self.beta) * loss_ce
                # else:
                #     loss = loss_ce

                if self.global_protos is not None:
                    proto_postive = copy.deepcopy(rep.detach())
                    negative_protos_list = []
                    for i, yy in enumerate(y):# 遍历每个样本
                        y_c = yy.item()#查看该样本标签
                        if type(self.global_protos[y_c]) != type([]):
                            proto_postive[i, :] = self.global_protos[y_c].detach()#收集与该样本标签一致的proto作为正样本proto

                        neg_proto_dict = {key: data for key, data in self.global_protos.items() if key != y_c}
                        neg_proto_list = [torch.tensor(proto) for proto in neg_proto_dict.values()]
                        # 把剩下的负proto拼成一个大的Tensor
                        neg_proto = torch.stack(neg_proto_list)
                        negative_protos_list.append(neg_proto)

                    negative_protos = torch.stack(negative_protos_list)

                    sim_pos = F.cosine_similarity(rep, proto_postive, dim=-1) / self.temp
                    negative_similariy_proto = F.cosine_similarity(rep.unsqueeze(1), negative_protos, dim=-1)/ self.temp
                    sim_neg = negative_similariy_proto

                    labels = torch.zeros(self.batch_size, dtype=torch.long).to(self.device)
                    logit = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=-1).to(self.device)

                    CP_loss = F.cross_entropy(logit, labels)
                    P_loss = self.loss_mse(proto_postive, rep)
                    loss_1 = self.beta * torch.mean(CP_loss) + (1-self.beta) * P_loss
                    loss = CE_loss + loss_1
                else:
                    loss = CE_loss
                    
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_num += y.shape[0]
                total_losses += loss.item() * y.shape[0]

        self.total_train_num = total_train_num
        self.total_losses = total_losses / self.local_epochs

        # 更新old_model
        # self.old_model = copy.deepcopy(self.model)


        self.collect_protos_and_weights()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)


    def collect_protos_and_weights(self):
        self.model.eval()

        protos = defaultdict(list)
        confidence = defaultdict(list)

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
                output = self.model.head(rep)
                softmax_out = F.softmax(output, dim=1)

                for i, yy in enumerate(y):
                    # 收集每个类的proto
                    y_c = yy.item()
                    if self.is_ws:
                        protos[y_c].append((softmax_out[i, y_c].detach().data, rep[i, :].detach().data))
                        confidence[y_c].append(softmax_out[i, y_c].detach().data) #根据类标签收集每个样本的置信度
                    else:
                        protos[y_c].append(rep[i, :].detach().data)

            if self.is_ws:
                for j in range(self.num_classes):
                    if len(confidence[j]) == 0:  # 如果该类的置信度个数为0，说明本地没有这个标签，直接赋值为0
                        self.confidence_w[j] = 0
                    else:
                        self.confidence_w[j] = sum(confidence[j]) / len(confidence[j])

        if self.is_ws:
            self.protos = agg_func_con(protos)
        else:
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


def agg_func_con(protos):
    """
    Returns the average of the weights.
    """

    for [label,  w_proto_list] in protos.items():
        if len(w_proto_list) > 1:
            proto_total = 0 * w_proto_list[0][1].data
            w_total = 0
            for (w, proto) in w_proto_list:
                proto_total += proto.data * w
                w_total += w
            protos[label] = proto_total / w_total
        else:
            protos[label] = w_proto_list[0][1]

    return protos

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