import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
# from system.utils.privacy import *


class clientAVG(Client):
    def __init__(self, args, id, model_heter, stride, train_samples, test_samples, **kwargs):
        super().__init__(args, id, model_heter, stride, train_samples, test_samples, **kwargs)

    def train(self):
        self.model.train()
        
        start_time = time.time()

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
                output = self.model(x)
                loss = self.loss(output, y)
                # nn.CrossEntropyLoss 的输入通常是模型的未经 softmax 处理的 logits。模型的输出概率分布会在损失函数内部计算
                # nn.CrossEntropyLoss 在内部计算了 softmax 概率分布，并在计算损失时将其与真实标签进行比较。它同时包括了 softmax 和负对数似然（NLL）损失的计算。
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_train_num += y.shape[0]
                total_losses += loss.item() * y.shape[0]

        self.total_train_num = total_train_num
        self.total_losses = total_losses / self.local_epochs

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        return self.total_losses, self.total_train_num