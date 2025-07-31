import pickle


def Find_categories(dataset, ori_env, method, threshold):
    all_categories = []

    for day in range(7):
        with open(f'IoT_data/{dataset}/{ori_env}/trn_day_{day}_{method}_th={threshold}.pkl', 'rb') as file3:
            data = pickle.load(file3)
        if len(data) <= 30:
            all_categories.append(day)
        else:

            num_subgroups = (len(data) + 29) // 30
            for i in range(num_subgroups):
                start_idx = i * 30
                end_idx = min((i + 1) * 30, len(data))
                subgroup = data[start_idx:end_idx]

                subgroup_name = f"{day}_{i}"
                filename = f'IoT_data/{dataset}/{ori_env}/trn_day_{subgroup_name}_{method}_th={threshold}.pkl'

                with open(filename, 'wb') as f:
                    pickle.dump(subgroup, f)

                all_categories.append(subgroup_name)

    return all_categories


def Find_categories_increase(dataset, new_env, method, threshold, model):
    all_categories = []

    with open(
            f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.pkl',
            'rb') as file3:
        data = pickle.load(file3)

    num_subgroups = (len(data) + 19) // 20
    for i in range(num_subgroups):
        start_idx = i * 20
        end_idx = min((i + 1) * 20, len(data))
        subgroup = data[start_idx:end_idx]

        subgroup_name = i
        filename = f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{subgroup_name}.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(subgroup, f)

        all_categories.append(subgroup_name)

    return all_categories


def Find_categories_filter(dataset, new_env, method, threshold, model):
    all_categories = []

    with open(
            f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_seq_increase.pkl',
            'rb') as file3:
        data = pickle.load(file3)

    num_subgroups = (len(data) + 29) // 30
    for i in range(num_subgroups):
        start_idx = i * 29
        end_idx = min((i + 1) * 30, len(data))
        subgroup = data[start_idx:end_idx]

        subgroup_name = i
        filename = f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{subgroup_name}.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(subgroup, f)

        all_categories.append(subgroup_name)

    return all_categories
