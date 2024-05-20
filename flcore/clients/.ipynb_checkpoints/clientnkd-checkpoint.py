import torch
import torch.nn as nn
import numpy as np
import time
import copy
import torch.nn.functional as F
from flcore.clients.clientbase import Client

# from system.utils.privacy import *
from collections import defaultdict



class clientNKD(Client):
    def __init__(self, args, id, train_samples, test_samples, noise_loader, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.noise_loader = noise_loader
        # self.global_model = copy.deepcopy(args.model)  # 客户端用来存放全局模型参数的模型

        # self.sample_per_class存放的是该客户端的训练集中每一个类对应的样本数量
        self.sample_per_class = torch.zeros(self.num_classes)# 用来存放该客户端每一类的样本数，初始是一个大小为num_class的tensor
        trainloader = self.load_train_data()# 读取自己客户端的数据集
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        # tensor.item()：将张量中的单个元素转换为Python标量
        self.qualified_labels = []

        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)# 默认也算的是均值
        # self.discrimin_loss = nn.BCELoss().to(self.device) #默认算的是均值 # nn.BCELoss()是PyTorch中的二元交叉熵损失函数，全称为Binary Cross Entropy Loss。它通常用于二分类问题，比如GAN中的判别器训练阶段。
        self.KL_loss = nn.KLDivLoss(reduction="batchmean").to(self.device)
        self.MSE = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma)

        # self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        # self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(self.device)
        # self.optimizer_W = torch.optim.SGD(self.W_h.parameters(), lr=self.learning_rate)
        # self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer_W,
        #     gamma=args.learning_rate_decay_gamma
        # )

        self.logits = None
        # self.global_logits = None

        self.ll_lamda = args.logit_loss_lamda
        # self.dl_lamda = args.distill_loss_lamda
        # self.hl_lamda = args.hidden_loss_lamda

        self.noise_logit = []



    def train(self, is_nkd):
        # 在服务器将全局模型送给本地之后，本地要更新模型参数
        # 除了batchnorm层之外的其他层等于global_model的参数
        # self.update_local_bn()
        self.model.to(self.device)
        self.model.train()
        self.trainloader = self.load_train_data()
        start_time = time.time()
        max_local_epochs = self.local_epochs#本地训练的epoch
        logits = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(self.trainloader):
                loss = 0
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output_l = self.model(x)
                CE_loss = self.loss(output_l, y)
                loss += CE_loss

                # 先用最简单的试试
                
                # if self.global_logits != None:
                #     logit_new = copy.deepcopy(output_l.detach())
                #     for i, yy in enumerate(y):
                #         y_c = yy.item()
                #         if type(self.global_logits[y_c]) != type([]):
                #             logit_new[i, :] = self.global_logits[y_c].data
                #     loss += self.MSE(logit_new, output_l) * self.ll_lamda
                #
                # for i, yy in enumerate(y):
                #     y_c = yy.item()
                #     logits[y_c].append(output_l[i, :].detach().data)

                # if is_nkd:
                #     self.global_model.eval()
                #     rep_g = self.global_model.base(x)
                #     output_g = self.global_model.head(rep_g)
                #     CE_loss_g = self.loss(output_g, y)
                #
                #     # distill_loss = self.KL_loss(F.log_softmax(result_local, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                #     distill_loss = self.DistillationLoss(output_l, output_g) / (CE_loss + CE_loss_g)
                #     # hidden_loss = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)
                #     loss += distill_loss
                #     # loss +=  distill_loss + hidden_loss

                self.optimizer.zero_grad()
                # if is_nkd:
                #     self.optimizer_W.zero_grad()
                loss.backward()
                # loss.backward(retain_graph=True)
                self.optimizer.step()
                # if is_nkd:
                #     self.optimizer_W.step()

        if self.learning_rate_decay:
            self.lr_scheduler.step()
            # self.learning_rate_scheduler_W.step()
            # 在测试损失停止改善的情况下要等待多少个 epoch（训练轮次）才会降低学习率。如果在 10 个 epoch 中验证损失没有显著改善，学习率将被降低。

        self.logits = agg_func(logits)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time



    def get_noise_logits(self):
        self.model.to(self.device)
        self.model.train()
        for i, n in enumerate(self.noise_loader):
            if type(n) == type([]):
                n[0] = n[0].to(self.device)
            else:
                n = n.to(self.device)
            with torch.no_grad():
                logit = self.model(n)
            self.noise_logit.append(logit)
        return self.noise_logit

    def get_bn_base_ldata(self):
        self.model.to(self.device)
        self.model.train()
        for i, (x,_) in enumerate(self.trainloader):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            with torch.no_grad():
                output = self.model(x)

        # 收集全局模型经过本地数据集之后的batchnorm层参数
        bn_params_base_local = {}
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                bn_params_base_local[name] = {
                    'mean': module.running_mean.clone(),
                    'var': module.running_var.clone()
                }
        return bn_params_base_local


    def set_model_param(self, global_model):
        # 1. 获取源模型的参数
        global_params = dict(global_model.named_parameters())
        global_bn_params = dict(global_model.named_buffers())

        # 2. 将参数应用到全局模型
        for name, param in self.model.named_parameters():
            if name in global_params:
                param.data.copy_(global_params[name].data)

        # 3. 处理BatchNorm的running_mean和running_var
        for name, buffer in self.model.named_buffers():
            if name in global_bn_params:
                buffer.data.copy_(global_bn_params[name].data)

        # 现在，self.global_model具有与global_model相同的参数和BatchNorm统计信息

        # global_model.parameters()访问的是模型可学习的参数，而batchnorm层中的running_mean和running_var是模型的统计信息，使用该函数，不能有效赋值

    def set_global_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)

    def DistillationLoss(self,student, teacher, T=1):
        student = F.log_softmax(student / T, dim=-1)
        teacher = (teacher / T).softmax(dim=-1)

        return -(teacher * student).sum(dim=1).mean()

def agg_func(logits):
    """
    Returns the average of the weights.
    """

    for [label, logit_list] in logits.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            logits[label] = logit / len(logit_list)
        else:
            logits[label] = logit_list[0]

    return logits


