#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.servers.serverlocal import Local
from flcore.servers.serverfomo import FedFomo
# Traditional FL
from flcore.servers.serveravg import FedAvg
# Regularization-based FL
from flcore.servers.serverprox import FedProx
from flcore.servers.serverdyn import FedDyn
# Model-splitting-based FL

# 这个用不了
from flcore.servers.serverscaffold import SCAFFOLD

from flcore.servers.servermoon import MOON
# Knowledge-distillation-based FL
from flcore.servers.servergen import FedGen
from flcore.servers.serverproto import FedProto
from flcore.servers.serverkd import FedKD
from flcore.servers.serverper import FedPer
from flcore.servers.serverrep import FedRep

# Knowledge-distillation-based pFL
from flcore.servers.serverdistill import FedDistill


from flcore.servers.serverCP import FedCP

from flcore.servers.serverMP import FedMP

from flcore.servers.serverCP2 import FedCP2

from flcore.trainmodel.models import *

from flcore.trainmodel.resnet import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

import torch.utils.model_zoo as model_zoo

logger = logging.getLogger()
# 创建一个日志记录器对象logger,logging.getLogger()返回一个默认的根记录器对象，用于处理日志消息
logger.setLevel(logging.ERROR)
#  设置日志记录器的日志级别为ERROR。这意味着只有ERROR级别的日志消息会被记录，其他级别的日志消息将被忽略。
#  这样可以控制日志记录的详细程度，只记录重要的错误信息。

warnings.simplefilter("ignore")
# 设置警告过滤器，将警告信息忽略掉。通过这行代码，警告消息将不会被显示或打印出来。
# 警告通常用于指示潜在问题或不推荐使用的代码，但有时可能会干扰到程序的执行。
torch.manual_seed(0)


# hyper-params for Text tasks
vocab_size = 98635
max_len=200
emb_dim=32



def run(args):

    time_list = []
    reporter = MemReporter() # 内存报报告器
    model_str = args.model

    for i in range(args.prev, args.times): # 其实就是训练几轮的意思
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        stride = [2, 2]
        # if args.dataset == "Cifar10":
        #     resnet = resnet18(stride, pretrained=False, num_classes=args.num_classes)
        #     initial_weight = model_zoo.load_url(model_urls['resnet18'])
        #     model = resnet
        #     initial_weight_1 = model.state_dict()
        #     for key in initial_weight.keys():
        #         if key[0:3] == 'fc.' or key[0:5]=='conv1' or key[0:3]=='bn1':
        #             initial_weight[key] = initial_weight_1[key]
        #
        #     model.load_state_dict(initial_weight)

        if args.dataset == "Cifar10":
            args.model = resnet18(stride, pretrained=False, num_classes=args.num_classes).to(args.device)
        elif args.dataset == "Cifar100" or args.dataset == "Tiny_imagenet":
            args.model = resnet50(stride, pretrained=False, num_classes=args.num_classes).to(args.device)
        else:
            raise NotImplementedError

        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)# Linear(in_features=512, out_features=10, bias=True)
            args.model.fc = nn.Identity()# 样做是为了将模型的最后一层全连接层（或称为头部）替换为一个恒等映射，从而将模型的输出直接作为特征表示，而不进行额外的变换或预测。
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "FedCPows":
            args.is_ws = False
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "FedMP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedMP(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(model_heter=args.model_heter, partition=args.partition, n=args.n, k=args.k,alpha=args.alpha, dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times, batch_size=args.batch_size, local_epochs=args.local_epochs,learning_rate = args.local_learning_rate,global_rounds=args.global_rounds)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--partition", type=str, default="dir",
                        help="非独立同分布的数据是pathological non-IID还是practical non-IID",
                        choices=["pat", "dir", "-"])
    parser.add_argument("--alpha", type=float, default=0.1, help="non-iid程度，越小代表异构程度越高，但是1代表iid", choices=[0.1, 0.5])
    parser.add_argument("--n", type=float, default=3, help="拥有的类数", choices=[3, 5, 8])
    parser.add_argument("--k", type=float, default=250, help="每个类的样本数")

    parser.add_argument('-data', "--dataset", type=str, default="Tiny_imagenet", choices= ["Cifar10", "Cifar100" , "Tiny_imagenet"])
    parser.add_argument('-nb', "--num_classes", type=int, default=200, choices=[10, 100, 200])
    parser.add_argument('-m', "--model", type=str, default="resnet")
    parser.add_argument('-mh', "--model_heter", type=bool, default=False)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001, help="Local learning rate", choices=[0.01, 0.005, 0.001])
    parser.add_argument('-algo', "--algorithm", type=str, default="FedGen")
    parser.add_argument('-ws', "--is_ws", type=str, default=True, help="是否使用置信度权重方案")
    
    # FedCP
    parser.add_argument('-T', "--temp", type=float, default=1.0, help="FedCP的对比损失里的温度参数")
    parser.add_argument('-bC', "--beta", type=float, default=0.6, help="FedCP的对比损失里的系数")

    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")

    # 本地训练参数
    parser.add_argument('-lbs', "--batch_size", type=int, default=40, help="local batch size")
    parser.add_argument('-ls', "--local_epochs", type=int, default=10,
                        help="Multiple update steps in one local epoch.")

    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=True)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.998)

    # 全局训练参数
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)

    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.5,
                        help="Ratio of clients per round，每一轮都选择固定数量(join_ratio*num_clients)的客户端去参加训练")


    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round，每一轮都选择随机个数的客户端去参加训练")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')

    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-fte', "--fine_tuning_epoch", type=int, default=0)

    # SCAFFOLD
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)

    # practical，模拟真实场景
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="The dropout rate for total clients. 被选中的客户将在每一轮培训中随机退出")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally.一旦一个客户被选为“slow trainers”，它就会比原来的训练器训练得慢")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    # FedProx
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")

    # MOON
    parser.add_argument('-ta', "--tau", type=float, default=1.0)

    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")

    # FedDistill
    parser.add_argument('-lamda', "--lamda", type=float, default=1.0)

    # FedRep
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)

    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=256)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.001)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=8192)
    parser.add_argument('-se', "--server_epochs", type=int, default=20)# 这个是在服务器里训练生成器的epoch
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False) # 这个参数用来控制特征提取器是否上传到服务器里，False代表上传，True代表不上传，只在本地更新

    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)

    



    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Alpha: {}".format(str(args.alpha)))
    print("Model Heter: {}".format(str(args.model_heter)))
    print("beta: {}".format(str(args.beta)))
    # print("k: {}".format(str(args.k)))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    # print("Using DP: {}".format(args.privacy))
    # if args.privacy:
    #     print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch))
    print("=" * 50)

    run(args)


