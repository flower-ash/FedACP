from flcore.clients.clientmoon import clientMOON
from flcore.servers.serverbase import Server
import time


class MOON(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientMOON)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i+1}-------------")
                print("\nEvaluate global models:")
                self.global_evaluate()
                print("\nEvaluate local models:")
                self.local_evaluate()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.local_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.local_test_acc))
        print("\nBest local accuracy.")
        print("\nAveraged time per iteration.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.save_local_model()

