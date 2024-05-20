from flcore.clients.clientCP import ClientCP
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time
import numpy as np
from collections import defaultdict
import torch
import h5py
import os


class FedCP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(ClientCP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")


        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = {key: torch.zeros(self.f_dim).to(self.device) for key in range(0, args.num_classes)}
        self.beta = args.beta
        self.is_ws = args.is_ws
        if self.is_ws:
            self.algorithm = self.algorithm
        else:
            self.algorithm = "FedCPows"
            
        



    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            for client in self.selected_clients:
                client.train()

            self.receive_protos()

            self.proto_aggregation()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i+1}-------------")
                print("\nEvaluate local models:")
                self.local_evaluate()

            self.send_param()


            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])


            if self.auto_break and self.check_done(acc_lss=[self.local_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.local_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_local_model()
        

    def send_param(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        confidence_w_list = []
        sample_per_class_list = []
        for client in self.selected_clients:

            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            sample_per_class_list.append(client.sample_per_class)
            confidence_w_list.append(client.confidence_w)

        uploaded_confidence_w = torch.stack( confidence_w_list, dim=0)
        col_sum = torch.sum(uploaded_confidence_w, dim=0)
        # 如果这一列的和为0，则self.uploaded_weights对应列为0
        self.confidence_weights = torch.where(col_sum == 0, torch.tensor(0.0), uploaded_confidence_w / col_sum)

        # 用数量作权重
        # 使用torch.stack将列表中的张量拼接成一个大的张量，sample_per_class这个张量里的每一行是一个客户端，每个数代表该客户端的标签中有该类的数量
        sample_per_class = torch.stack(sample_per_class_list, dim=0)
        # 活跃用户里的标签占比概率
        sum_per_class = torch.sum(sample_per_class, dim=0)  # 每个类的样本量
        # 每一个活跃用户里的y类对所有活跃用户里的y类的占比情况
        self.sample_per_class_prob = sample_per_class / sum_per_class

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct_l = []
        tot_correct_g = []
        for c in self.clients:  # 只测试活跃用户
            correct_l, correct_g, ns = c.test_metrics()  # ct:客户端测试正确的样本数；ns:客户端总的测试样本数；auc:客户端的auc
            tot_correct_l.append(correct_l * 1.0)
            tot_correct_g.append(correct_g * 1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct_l, tot_correct_g

    def local_evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()  # ids, num_samples, tot_correct_l, tot_correct_g
        stats_train = self.train_metrics()  # ids, num_samples, losses

        test_acc_l = sum(stats[2]) * 1.0 / sum(stats[1])
        test_acc_g = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs_l = [a / n for a, n in zip(stats[2], stats[1])]
        accs_g = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.local_test_acc.append((test_acc_l, test_acc_g))
        else:
            acc.append((test_acc_l, test_acc_g))

        if loss == None:
            self.local_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        self.local_test_acc_std.append((np.std(accs_l), np.std(accs_g)))

        print("客户端平均训练损失: {:.4f}".format(train_loss))
        print("客户端正确率: {:.4f}".format(test_acc_l))
        print("客户端protos正确率: {:.4f}".format(test_acc_g))
        print("正确率标准差: {:.4f}".format(np.std(accs_l)))
        print("proto正确率标准差: {:.4f}".format(np.std(accs_g)))
            

    def proto_aggregation(self):
        agg_protos_label = defaultdict(list)
        for i, local_protos in enumerate(self.uploaded_protos):#i：第i个客户端 local_protos：第i个客户端的proto集合
            for label in local_protos.keys():
                if self.is_ws:
                    weight = self.confidence_weights[i, label] # 用置信度作为权重
                    # weight = self.sample_per_class_prob[i, label] #用数量作为权重
                    agg_protos_label[label].append(weight * local_protos[label])
                else:
                    agg_protos_label[label].append(local_protos[label])

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) == 1:# 如果只收集到一个，就直接等于这个proto，如果等于0，不修改，用原来的proto
                self.global_protos[label] = proto_list[0].data
            elif len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                self.global_protos[label] = proto


    def save_results(self):
        if self.partition == "dir":
            algo = self.dataset + "_" + self.algorithm + "_a" + str(self.alpha) + "_lbs" + str(
                self.batch_size) + "_ls" + str(self.local_epochs) + "_lr" + str(self.learning_rate) + "_beta" + str(
                self.beta) + "_gr" + str(self.global_rounds)

        elif self.partition == "pat":
            algo = self.dataset + "_" + self.algorithm + "_n" + str(self.n) + "_k" + str(self.k) + "_lbs" + str(
                self.batch_size) + "_ls" + str(self.local_epochs) + "_lr" + str(self.learning_rate) + "_beta" + str(
                self.beta) + "_gr" + str(self.global_rounds)

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

    def save_local_model(self):
        if self.partition == "dir":
            file_name = self.dataset + "_a" + str(self.alpha) + "_lr" + str(self.learning_rate) + "_beta" + str(self.beta)
        elif self.partition == "pat":
            file_name = self.dataset + "_n" + str(self.n) + "_k" + str(self.k) + "_lr" + str(self.learning_rate) + "_beta" + str(self.beta)


        for client in self.clients:
            if self.model_heter: #如果是模型异构
                model_path = os.path.join("./client_models/model_heter", self.algorithm, file_name)
            else:
                model_path = os.path.join("./client_models/data_heter", self.algorithm, file_name)

            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, "client" + str(client.id) + ".pt")
            torch.save(client.model, model_path)