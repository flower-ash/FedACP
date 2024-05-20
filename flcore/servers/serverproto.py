from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import time
import numpy as np
from collections import defaultdict
import torch


class FedProto(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = {key: torch.zeros(self.f_dim).to(self.device) for key in range(0, args.num_classes)}


    def train(self):
        for i in range(self.global_rounds):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            for client in self.selected_clients:
                client.train()

            self.receive_protos()

            self.proto_aggregation(self.uploaded_protos)

            # 在聚合之后，传模型之前评估模型，免得本地模型被更新
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i + 1}-------------")
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
        

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, self.num_clients, self.partition, self.alpha, i, self.n, self.k, is_train=True)
            test_data = read_client_data(self.dataset, self.num_clients, self.partition, self.alpha, i, self.n, self.k, is_train=False)
            if self.model_heter:
                if i < 10:
                    stride = [1, 4]
                else:
                    stride = [2, 2]
            else:
                stride = [2, 2]
            client = clientObj(self.args,
                            id=i,
                            model_heter = self.model_heter,
                            stride = stride,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=train_slow,
                            send_slow=send_slow)
            self.clients.append(client)


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
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)


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
        stats = self.test_metrics()# ids, num_samples, tot_correct_l, tot_correct_g
        stats_train = self.train_metrics()# ids, num_samples, losses

        test_acc_l = sum(stats[2])*1.0 / sum(stats[1])
        test_acc_g = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
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
            

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
    def proto_aggregation(self,local_protos_list):
        agg_protos_label = defaultdict(list)
        for local_protos in local_protos_list:
            for label in local_protos.keys():
                agg_protos_label[label].append(local_protos[label])

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) == 1:  # 如果只收集到一个，就直接等于这个proto，如果等于0，不修改，用原来的proto
                self.global_protos[label] = proto_list[0].data
            elif len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                # 此处可以用不同的权重聚合proto
                self.global_protos[label] = proto / len(proto_list)

