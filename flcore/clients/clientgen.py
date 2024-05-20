import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
# from system.utils.privacy import *


class clientGen(Client):
    def __init__(self, args, id, model_heter, stride, train_samples, test_samples, **kwargs):
        super().__init__(args, id, model_heter, stride, train_samples, test_samples, **kwargs)

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad(): # 使用torch.no_grad()进行推断阶段，禁止梯度跟踪
                rep = self.model.base(x).detach()# 输出的是特征
            break
        self.feature_dim = rep.shape[1]

        self.sample_per_class = torch.zeros(self.num_classes)# 用来存放该客户端每一类的样本数
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1
        # tensor.item()：将张量中的单个元素转换为Python标量
        self.qualified_labels = []
        self.generative_model = None
        self.localize_feature_extractor = args.localize_feature_extractor
        

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        # if self.privacy:
        #     self.model, self.optimizer, trainloader, privacy_engine = \
        #         initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                
                labels = np.random.choice(self.qualified_labels, self.batch_size)#随机抽取一个batch的标签y
                labels = torch.LongTensor(labels).to(self.device)
                z = self.generative_model(labels)# 把随机抽取的标签输入到生成器
                loss += self.loss(self.model.head(z), labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # if self.privacy:
        #     eps, DELTA = get_dp_params(privacy_engine)
        #     print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
            
        
    def set_parameters(self, model, generative_model, qualified_labels):
        if self.localize_feature_extractor:
            for new_param, old_param in zip(model.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(model.parameters(), self.model.parameters()):# 把全局模型参数model.parameters()送到客户端里
                old_param.data = new_param.data.clone()

        self.generative_model = generative_model
        self.qualified_labels = qualified_labels

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
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
                
                labels = np.random.choice(self.qualified_labels, self.batch_size)
                labels = torch.LongTensor(labels).to(self.device)
                z = self.generative_model(labels)
                loss += self.loss(self.model.head(z), labels)
                
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
