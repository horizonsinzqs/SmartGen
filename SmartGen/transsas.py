import pickle


def Transsas(dataset, new_env, threshold, method, model):
    with open(
            f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_filter_true.pkl',
            'rb') as file2:
        sequence = pickle.load(file2)
    actions = sequence

    with open(f'data/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.txt', 'w') as file:
        pass

    for t in range(len(actions)):

        id_list = t
        behavior_list = actions[t]

        with open(f'data/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.txt', 'a') as file:

            for i in range(len(behavior_list)):
                file.write(f"{t} {behavior_list[i]}\n")


def Transsas_baseline(dataset, ori_env):
    with open(f'IoT_data/{dataset}/{ori_env}/trn.pkl', 'rb') as file2:
        sequence = pickle.load(file2)
    actions = sequence

    with open(f'data/{dataset}_{ori_env}_trn.txt', 'w') as file:
        pass

    for t in range(len(actions)):

        id_list = t
        behavior_list = actions[t]

        with open(f'data/{dataset}_{ori_env}_trn.txt', 'a') as file:

            for i in range(len(behavior_list)):
                file.write(f"{t} {behavior_list[i]}\n")


def Transsas_testdata(dataset, new_env):
    with open(f'IoT_data/{dataset}/{new_env}/split_test.pkl', 'rb') as file2:
        sequence = pickle.load(file2)
    actions = sequence

    with open(f'data/{dataset}_{new_env}_split_test.txt', 'w') as file:
        pass

    for t in range(len(actions)):

        id_list = t
        behavior_list = actions[t]

        with open(f'data/{dataset}_{new_env}_split_test.txt', 'a') as file:

            for i in range(len(behavior_list)):
                file.write(f"{t} {behavior_list[i]}\n")
