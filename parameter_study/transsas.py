import pickle


def Transsas_EP(dataset, new_env, threshold, method, model, study):
    with open(f'study_data/{study}/{dataset}_{new_env}_generation_SPPC_th={threshold}_{model}_seq_filter_true.pkl',
              'rb') as file2:
        sequence = pickle.load(file2)
    actions = sequence

    with open(f'data/{study}_{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.txt', 'w') as file:
        pass

    for t in range(len(actions)):
        id_list = t
        behavior_list = actions[t]

        with open(f'data/{study}_{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.txt', 'a') as file:
            for i in range(len(behavior_list)):
                file.write(f"{t} {behavior_list[i]}\n")

    print("Data has been written to file.")


def Transsas_splitEP(dataset, new_env, threshold, method, model, study, sp1, sp2):
    with open(
            f'study_data/{study}/{dataset}_{new_env}_generation_SPPC_th={threshold}_{model}_seq_filter_true_{sp1}_{sp2}.pkl',
            'rb') as file2:
        sequence = pickle.load(file2)
    actions = sequence

    with open(f'data/{study}_{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_{sp1}_{sp2}.txt',
              'w') as file:
        pass

    for t in range(len(actions)):
        id_list = t
        behavior_list = actions[t]

        with open(f'data/{study}_{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_{sp1}_{sp2}.txt',
                  'a') as file:
            for i in range(len(behavior_list)):
                file.write(f"{t} {behavior_list[i]}\n")

    print("Data has been written to file.")


def Transsas_test(dataset, new_env):
    with open(f'test/{dataset}/{new_env}/split_test.pkl', 'rb') as file2:
        sequence = pickle.load(file2)
    actions = sequence

    with open(f'data/{dataset}_{new_env}_test.txt', 'w') as file:
        pass

    for t in range(len(actions)):
        id_list = t
        behavior_list = actions[t]

        with open(f'data/{dataset}_{new_env}_test.txt', 'a') as file:
            for i in range(len(behavior_list)):
                file.write(f"{t} {behavior_list[i]}\n")

    print("Data has been written to file.")
