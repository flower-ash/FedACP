import torch
import os
import numpy as np
import h5py
import copy
import time
import random

from utils.data_utils import read_client_data
from utils.dlg import DLG

import torchvision
import torchvision.transforms as transforms
from flcore.trainmodel.ImageFolder_custom import ImageFolder_custom


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate

        self.partition = args.partition
        self.alpha = args.alpha
        self.n = args.n
        self.k = args.k

        self.global_model = copy.deepcopy(args.model)

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break


        self.clients = [] # 里面放了所有的客户端对象
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.local_test_acc = [] # 在测试集的精度
        self.local_test_auc = [] # 在测试集的auc
        self.local_train_loss = [] # 在训练集的损失
        self.global_test_acc = [] # 全局模型的精度测试
        self.local_test_acc_std = [] # 测试集精度的STD

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch = args.fine_tuning_epoch

        self.model_heter = args.model_heter

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        if self.dataset == "Cifar10":
            testset = torchvision.datasets.CIFAR10(root='/root/autodl-tmp/DATA', train=False, download=False, transform=transform_test)
        elif self.dataset == "Cifar100":
            testset = torchvision.datasets.CIFAR100(root='/root/autodl-tmp/DATA', train=False, download=False, transform=transform_test)
        elif self.dataset == "Tiny_imagenet":
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            testset = ImageFolder_custom(root="/root/autodl-tmp/DATA/tiny-imagenet-200/val", transform=transform)

        self.global_testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        if self.dataset == "Cifar10":
            self.f_dim = 512
        elif self.dataset == "Cifar100" or self.dataset == "Tiny_imagenet":
            self.f_dim = 2048

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, self.num_clients, self.partition, self.alpha, i, self.n, self.k, is_train=True)
            test_data = read_client_data(self.dataset, self.num_clients, self.partition, self.alpha, i, self.n, self.k, is_train=False)
            if self.model_heter:
                if i < 7:
                    stride = [1, 4]
                elif i>=7 and i<14:
                    stride = [1, 2]
                else:
                    stride = [2, 2]
            else:
                stride = [2, 2]
            client = clientObj(self.args,
                               id=i,
                               model_heter=self.model_heter,
                               stride=stride,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)



    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.selected_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        # 初始化全局模型
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("./global_models",  self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm +"_a" + str(self.alpha) +"_lbs"+str(self.batch_size)+"_ls"+str(self.local_epochs)+"_lr"+str(self.learning_rate)+"_gr"+str(self.global_rounds)+ "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def save_local_model(self):
        if self.partition == "dir":
            file_name = self.dataset + "_a" + str(self.alpha) + "_lr" + str(self.learning_rate)
        elif self.partition == "pat":
            file_name = self.dataset + "_n" + str(self.n) + "_k" + str(self.k) + "_lr" + str(self.learning_rate)


        for client in self.clients:
            if self.model_heter: #如果是模型异构
                model_path = os.path.join("./client_models/model_heter", self.algorithm,file_name)
            else:
                model_path = os.path.join("./client_models/data_heter", self.algorithm,
                                          self.dataset + "_a" + str(self.alpha) + "_lr" + str(self.learning_rate))

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, "client" + str(client.id) + ".pt")
            torch.save(client.model, model_path)

    def load_model(self):
        model_path = os.path.join("/models", self.dataset)
        model_path = os.path.join(model_path,
                                  self.algorithm + "_n" + str(self.n)+"_k" + str(self.k) + "_lbs" + str(self.batch_size) + "_ls" + str(
                                      self.local_epochs) + "_lr" + str(self.learning_rate) + "_gr" + str(
                                      self.global_rounds) + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        if self.partition == "dir":
            algo = self.dataset + "_" + self.algorithm + "_a" + str(self.alpha) + "_lbs" + str(
                self.batch_size) + "_ls" + str(self.local_epochs) + "_lr" + str(self.learning_rate) + "_gr" + str(
                self.global_rounds)

        elif self.partition == "pat":
            algo = self.dataset + "_" + self.algorithm + "_n" + str(self.n) + "_k" + str(self.k) + "_lbs" + str(
                self.batch_size) + "_ls" + str(self.local_epochs) + "_lr" + str(self.learning_rate) + "_gr" + str(
                self.global_rounds)

        if self.model_heter:
            result_path = "./results/model_heter/"
        else:
            result_path = "./results/data_heter/"

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.local_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('local_test_acc', data=self.local_test_acc)
                hf.create_dataset('local_test_acc_std', data=self.local_test_acc_std)
                hf.create_dataset('local_train_loss', data=self.local_train_loss)
                hf.create_dataset('global_test_acc', data=self.global_test_acc)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients: # 只测试活跃用户
            ct, ns, auc = c.test_metrics() # ct:客户端测试正确的样本数；ns:客户端总的测试样本数；auc:客户端的auc
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.selected_clients:# 只测试活跃用户
            cl, ns = c.train_metrics()#cl:客户端总损失；ns:客户端总样本数
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def local_evaluate(self, acc_l=None, loss_l=None):
        stats = self.test_metrics() # ids:所有客户端id, num_samples:每个客户端拥有的测试集样本数, tot_correct:每个客户端在测试集上预测正确的样本数, tot_auc:每个客户端在测试集上的auc
        stats_train = self.train_metrics() # ids, num_samples:每个客户端拥有的训练集样本数, losses::每个客户端在训练集上的损失

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc_l == None:
            self.local_test_acc.append(test_acc)
        else:
            acc_l.append(test_acc)
        
        if loss_l == None:
            self.local_train_loss.append(train_loss)
        else:
            loss_l.append(train_loss)

        print("客户端平均训练损失： {:.4f}".format(train_loss))
        print("客户端平均测试精度： {:.4f}%".format(100*test_acc))
        print("客户端平均测试AUC： {:.4f}".format(test_auc))
        print("客户端测试精度的标准差： {:.4f}".format(np.std(accs)))# 计算20个客户端的精度的标准差
        print("客户端测试AUC标准差： {:.4f}".format(np.std(aucs))) # 计算20个客户端的AUC的标准差


    def global_evaluate(self, acc_g=None):
        correct_global, total, global_loss = 0, 0, 0
        self.global_model.eval()

        for data in self.global_testloader:
            images, labels = data
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = self.global_model(images)
            total += labels.size(0)
            correct_global += torch.sum(torch.argmax(torch.softmax(outputs, dim=1), dim=1) == labels).item()

        test_acc_g = correct_global / total
        if acc_g == None:
            self.global_test_acc.append(test_acc_g)
        else:
            acc_g.append(test_acc_g)
        print("全局模型测试精度： {:.4f}%".format(100 * test_acc_g))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, self.num_clients, self.partition, self.alpha, i, self.n, self.k, is_train=True)
            test_data = read_client_data(self.dataset, self.num_clients, self.partition, self.alpha, i, self.n, self.k, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
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

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.local_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.local_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
