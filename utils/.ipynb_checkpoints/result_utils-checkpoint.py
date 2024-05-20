import h5py
import numpy as np
import os


def average_data(model_heter=False, partition="dir", n=3, k=250, alpha="", algorithm="", dataset="", goal="", times=10, batch_size = 50,local_epochs =10,learning_rate = 0.05,global_rounds=300):
    test_acc = get_all_results_for_one_algo(model_heter, partition, n, k, alpha, algorithm, dataset, goal, times,batch_size,local_epochs,learning_rate,global_rounds)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))

def get_all_results_for_one_algo(model_heter=False, partition="", n=3, k=250, alpha="", algorithm="", dataset="", goal="", times=10, batch_size = 50,local_epochs =10,learning_rate = 0.05,global_rounds=300):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        if partition == "dir":
            file_name = dataset + "_" + algorithms_list[i] + "_a" + str(alpha) + "_lbs" + str(batch_size) + "_ls" + str(
                local_epochs) + "_lr" + str(learning_rate) + "_gr" + str(global_rounds) + "_" + str(goal) + "_" + str(i)
        elif partition == "pat":
            file_name = dataset + "_" + algorithms_list[i] + "_n" + str(n) + "_k" + str(k) + "_lbs" + str(
                batch_size) + "_ls" + str(local_epochs) + "_lr" + str(learning_rate) + "_gr" + str(global_rounds) + str(
                goal) + "_" + str(i)

        test_acc.append(np.array(read_data_then_delete(model_heter, file_name, delete=False)))
    return test_acc



def read_data_then_delete(model_heter, file_name, delete=False):
    if model_heter:
        file_path = "./results/model_heter/" + file_name + ".h5"
    else:
        file_path = "./results/data_heter/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        local_test_acc = np.array(hf.get('local_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(local_test_acc))

    return local_test_acc