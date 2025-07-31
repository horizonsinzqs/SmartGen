import pickle


def Dayse(dataset, ori_env):
    with open(f'IoT_data/{dataset}/{ori_env}/split_trn.pkl', 'rb') as file3:
        data = pickle.load(file3)

    categories = {i: [] for i in range(7)}

    for sequence in data:
        first_element = sequence[0]
        if 0 <= first_element <= 6:
            categories[first_element].append(sequence)

    for key, value in categories.items():
        l = len(value)
        print(f"Class {key}: {l}")

    for key, value in categories.items():
        filename = f'IoT_data/{dataset}/{ori_env}/trn_day_{key}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(value, f)


def Dayse_increase(dataset, new_env, method, threshold, model):
    with open(
            f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.pkl',
            'rb') as file3:
        data = pickle.load(file3)

    categories = {i: [] for i in range(7)}

    for sequence in data:
        first_element = sequence[0]
        if 0 <= first_element <= 6:
            categories[first_element].append(sequence)

    for key, value in categories.items():
        l = len(value)
        print(f"Class {key}: {l}")

    for key, value in categories.items():
        filename = f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{key}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(value, f)


def Dayse_filter(dataset, new_env, method, threshold, model):
    with open(
            f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_seq_all.pkl',
            'rb') as file3:
        data = pickle.load(file3)

    categories = {i: [] for i in range(7)}

    for sequence in data:
        first_element = sequence[0]
        if 0 <= first_element <= 6:
            categories[first_element].append(sequence)

    for key, value in categories.items():
        l = len(value)
        print(f"Class {key}: {l}")

    for key, value in categories.items():
        filename = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{key}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(value, f)
