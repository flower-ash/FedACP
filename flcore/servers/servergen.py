import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientgen import clientGen
from flcore.servers.serverbase import Server
from threading import Thread


class FedGen(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGen)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = [] # 耗费的时间

        # self.fc_dim = args.fc_dim
        self.generative_model = Generative(
                                    args.noise_dim,
                                    args.num_classes, 
                                    args.hidden_dim, 
                                    self.clients[0].feature_dim, 
                                    self.device
                                ).to(self.device)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=args.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.generative_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=args.learning_rate_decay_gamma)
        self.loss = nn.CrossEntropyLoss()
        
        self.qualified_labels = [] # 含所有客户端数据集的类别标签
        for client in self.clients:
            for yy in range(self.num_classes):
                self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])

        self.server_epochs = args.server_epochs
        self.localize_feature_extractor = args.localize_feature_extractor
        if self.localize_feature_extractor:
            self.global_model = copy.deepcopy(args.model.head)

        

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            for client in self.selected_clients:# 选择的10个客户端进行训练
                client.train()

            self.receive_models() # 服务器接收来自客户端的信息
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.train_generator()#训练生成器
            self.aggregate_parameters() # 聚合本地模型

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i + 1}-------------")
                print("\nEvaluate global models:")
                self.global_evaluate()
                print("\nEvaluate local models:")
                self.local_evaluate()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.local_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.local_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_global_model()
        # self.save_local_model()


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model, self.generative_model, self.qualified_labels)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        # random.sample：用于从一个序列（列表、元组、字符串等可迭代对象）中随机选择指定数量的元素，返回一个新的列表。
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []# 记录客户端id
        self.uploaded_weights = []# 记录客户端的样本数占总的活跃用户样本数的比例，作为聚合的权重
        self.uploaded_models = []
        tot_samples = 0 # 记录总的样本数
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                if self.localize_feature_extractor:
                    self.uploaded_models.append(client.model.head)
                else:
                    self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def train_generator(self):
        self.generative_model.train()

        for _ in range(self.server_epochs):
            labels = np.random.choice(self.qualified_labels, self.batch_size)# 采样标签
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_model(labels)# 将标签输入到生成器

            logits = 0
            for w, model in zip(self.uploaded_weights, self.uploaded_models):
                model.eval()
                if self.localize_feature_extractor:
                    logits += model(z) * w
                else:
                    logits += model.head(z) * w # 将生成器输出z输入到每一个客户端的最后一个全连接层

            self.generative_optimizer.zero_grad()
            loss = self.loss(logits, labels)
            loss.backward()
            self.generative_optimizer.step()
        
        self.generative_learning_rate_scheduler.step()

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model, self.generative_model, self.qualified_labels)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim+num_classes, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device) # 从高斯分布中采样噪音
        y_input = torch.tensor(F.one_hot(labels, self.num_classes),dtype=torch.float32)
        # 拼接
        z = torch.cat((eps, y_input), dim=1)#把标签和噪音拼接起来
        # 前馈进网络
        z = self.fc1(z)
        z = self.fc(z)

        return z