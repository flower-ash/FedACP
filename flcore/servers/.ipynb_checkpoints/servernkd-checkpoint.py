import time
import torch
from flcore.clients.clientnkd import clientNKD
from flcore.servers.serverbase import Server
import torch.nn.functional as F
from torch.optim import Adam
import random
from collections import defaultdict

from threading import Thread
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import torchvision
import torchvision.transforms as transforms
from flcore.trainmodel.NoiseDataset import NoiseDataset
from utils.data_utils import read_client_data


class FedNKD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.epoch = args.FedNKD_server_epochs
        self.noise_len_batch = args.noise_len_batch
        self.noise_batch_size = args.noise_batch_size
        self.r = args.ft_set_rate
        self.CELoss = torch.nn.CrossEntropyLoss().to(self.device)
        self.glr = args.NKD_global_learning_rate
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_decay_gamma = args.learning_rate_decay_gamma
        self.ft_lr = args.fine_tuning_learning_rate

        # 采样噪音数据集，大小为self.len_batch * self.noise_batch_size
        noise_data = torch.randn(self.noise_len_batch * self.noise_batch_size, 3, 32, 32)
        noise_dataset = NoiseDataset(noise_data)
        self.noise_loader = torch.utils.data.DataLoader(noise_dataset, batch_size=self.noise_batch_size, shuffle=False)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientNKD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # global_logits的设置
        self.num_classes = args.num_classes
        self.global_logits = None
        self.nkd = True  # 是否进行无数据知识蒸馏
        self.is_finetune = args.is_finetune

    def train(self):

        for i in range(self.global_rounds):
            s_t = time.time()
            # 在每一轮训练结束之后测试
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i + 1}-------------")
                print("\nEvaluate local and global model")

            # if i > 49:  # 设置通信阈值，从第101轮开始，进行无数据知识蒸馏
            #     # if self.local_test_acc[i-1] > 0.6: # 设置精度阈值，当上一轮精度达到60%以上，这一轮可以开始进行无数据知识蒸馏
            #     self.nkd = True

            # 选择活跃用户
            self.selected_clients = self.select_clients()
            # 将global model、global logits以及noise dataloader送给本地
            self.send_to_client(self.global_model, self.global_logits)

            # 客户端在本地训练
            for client in self.selected_clients:
                client.train(self.nkd)
            # 训练完本地模型之后测试本地性能
            self.local_evaluate()
            # 接收local logit,noise_logit,id,weights
            self.receive_from_clients()
            # 聚合接收的local logit
            self.global_logits = logit_aggregation(self.uploaded_logits)

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            # 训练全局模型
            if self.nkd:  # 设置通信阈值
                # 每一轮重新设置优化器，放在前面的话，多轮之后learning rate太小，损失无法下降
                self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.glr)
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.learning_rate_decay_gamma)

                self.global_model.train()

                for j in range(self.epoch):
                    ND_loss = 0  # 每个epoch里的噪音损失
                    for i, noise_data in enumerate(self.noise_loader):
                        noise_data = noise_data.to(self.device)
                        loss = 0
                        student_output = self.global_model(noise_data)
                        for w, teacher_logits in zip(self.uploaded_weights, self.local_noise_logits):
                            teacher_output = teacher_logits[i]
                            loss_ = DistillationLoss(student_output , teacher_output) * w
                            loss += loss_
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        ND_loss += loss

                    if self.learning_rate_decay:
                        self.lr_scheduler.step()
                    ND_loss = ND_loss / (self.noise_len_batch)
                    if j % 10 == 0:
                        print("epoch{}:Global Model Noise Distillation Loss= {:.4f} ".format(j+1,ND_loss))

                # 将训练完成的global model传给本地
                self.sent_gm_to_client(self.global_model)
                # 收集经过本地数据集的global model的bn参数
                self.receive_bn()
                # 根据接收的bn参数更新全局模型的bn
                self.update_batchnorm()  # 更新batchnorm参数，让其等于所有客户端batchnorm层参数的均值
                acc_global, global_test_loss = self.global_evaluate()
                print("噪音蒸馏后的全局模型:")
                print("全局测试精度: {:.4f}%".format(100 * acc_global))
                print("全局测试损失: {:.4f}".format(global_test_loss))

                # # 用有限数据集进行微调，只有原始数据集的1%
                # self.train_limited_data()
                # acc_global, global_test_loss = self.global_evaluate()
                # print("用有限数据集对全局模型微调后:")
                # print("全局测试精度: {:.4f}%".format(100 * acc_global))
                # print("全局测试损失: {:.4f}".format(global_test_loss))

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.local_test_acc], top_cnt=self.top_cnt):
                break

        print("\n最好的本地精度：")
        print(max(self.local_test_acc))
        if self.nkd:  # 允许噪音蒸馏
            print("\n最好的全局精度：")
            print(max(self.global_test_acc))
            # if self.is_finetune:
            #     self.train_limited_data()
            #     acc_global, global_test_loss = self.global_evaluate()
            #     print("使用有限的数据集进行微调：")
            #     print("微调后的全局精度: {:.4f}%".format(100 * acc_global))
            #     print("微调后的测试损失: {:.4f}".format(global_test_loss))

        print("\n每轮通信平均耗时：")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, self.num_clients, self.alpha, i, is_train=True)
            test_data = read_client_data(self.dataset, self.num_clients, self.alpha, i, is_train=False)
            client = clientObj(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            noise_loader = self.noise_loader,
                            train_slow=train_slow,
                            send_slow=send_slow)
            self.clients.append(client)

    def send_to_client(self, global_model, global_logits):
        assert (len(self.selected_clients) > 0)
        # 送给所有客户端
        for client in self.selected_clients:
            start_time = time.time()
            if self.nkd:
                client.set_model_param(global_model)
                client.set_global_logits(global_logits)
            else:
                client.set_global_logits(global_logits)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += (time.time() - start_time)

    # 接收权重、模型、id以及logit
    def receive_from_clients(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_logits = []

        if self.nkd:
            self.uploaded_weights = []
            self.local_noise_logits = []

        tot_samples = 0
        for client in self.selected_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                # 接收本地id
                self.uploaded_ids.append(client.id)

                # 接收本地logit
                self.uploaded_logits.append(client.logits)

                if self.nkd:
                    # 接收权重
                    self.uploaded_weights.append(client.train_samples)

                    # 收集noise_logit
                    clients_noise_logits = client.get_noise_logits()
                    self.local_noise_logits.append(clients_noise_logits)

        if self.nkd:
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples

    def sent_gm_to_client(self,global_model):
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients: # 只需要送给活跃用户
            start_time = time.time()
            client.set_model_param(global_model)
            client.send_time_cost['total_cost'] += (time.time() - start_time)

    def receive_bn(self):
        assert (len(self.selected_clients) > 0)
        if self.nkd:
            self.global_bn_base_ldata = []
        for client in self.selected_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                if self.nkd:
                    # 收集本地模型的batchnorm层参数
                    bn_params_base_local = client.get_bn_base_ldata()
                    self.global_bn_base_ldata.append(bn_params_base_local)

    def update_batchnorm(self):

        # 更新全局模型batchnorm层的参数，使其等于本地模型该层的均值
        global_bn_params_avg = {}
        for bn, w in zip(self.global_bn_base_ldata, self.uploaded_weights):
            for name, bn_params in bn.items():
                if name in global_bn_params_avg:
                    global_bn_params_avg[name]['mean'] += bn_params['mean'] * w
                    global_bn_params_avg[name]['var'] += bn_params['var'] * w
                else:
                    global_bn_params_avg[name] = {
                        'mean': bn_params['mean'].clone() * w,
                        'var': bn_params['var'].clone() * w
                    }
        # 设置teacher的batchnorm层等于local_model的均值
        self.global_model.eval()
        for name, module in self.global_model.named_modules():
            if name in global_bn_params_avg and isinstance(module, torch.nn.BatchNorm2d):
                module.running_mean = global_bn_params_avg[name]['mean']
                module.running_var = global_bn_params_avg[name]['var']

    def train_limited_data(self):
        # 设置随机种子，以确保可重复性
        torch.manual_seed(42)

        # 定义数据转换
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # R,G,B每层的归一化用到的均值和方差
        ])
        if self.dataset == "Cifar10":
            train_dataset = torchvision.datasets.CIFAR10(root='/root/autodl-tmp/DataSet', train=True, download=False,
                                                         transform=transform_train)
        elif self.dataset == "Cifar100":
            train_dataset = torchvision.datasets.CIFAR100(root='/root/autodl-tmp/DataSet', train=True, download=False,
                                                          transform=transform_train)

        # 计算抽取的样本数
        total_samples = len(train_dataset)
        desired_samples = int(self.r * total_samples)

        # 随机抽取子集
        subset_indices = torch.randperm(total_samples)[:desired_samples]
        subset_sampler = SubsetRandomSampler(subset_indices)

        # 创建 DataLoader 使用 SubsetRandomSampler
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=subset_sampler)

        self.ft_optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.ft_lr)

        self.global_model.to(self.device)
        self.global_model.train()
        # 训练模型（示例，实际训练过程可能需要更多的配置和迭代次数）
        for epoch in range(30):  # 这里只训练10个epoch，你可能需要更多
            for inputs, labels in train_loader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                outputs = self.global_model(inputs)
                loss = self.CELoss(outputs, labels)
                self.ft_optimizer.zero_grad()
                loss.backward()
                self.ft_optimizer.step()
        

def DistillationLoss(student, teacher, T=1):
    student = F.log_softmax(student / T, dim=-1)
    teacher = (teacher / T).softmax(dim=-1)
    return -(teacher * student).sum(dim=1).mean()

def logit_aggregation(local_logits_list):
    agg_logits_label = defaultdict(list)
    for local_logits in local_logits_list:
        for label in local_logits.keys():
            agg_logits_label[label].append(local_logits[label])

    for [label, logit_list] in agg_logits_label.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            agg_logits_label[label] = logit / len(logit_list)
        else:
            agg_logits_label[label] = logit_list[0].data

    return agg_logits_label

