import pickle


def remove_quadruplets(nested_list):
    filtered_list = []
    for inner_list in nested_list:
        new_inner_list = []
        for i in range(0, len(inner_list), 4):
            quadruplet = inner_list[i:i + 4]
            if 99999 not in quadruplet:
                new_inner_list.extend(quadruplet)
        if len(new_inner_list) > 0:
            filtered_list.append(new_inner_list)
    return filtered_list


def Transnum(dataset, new_env, threshold, method, model, all_categories, dictionaries):
    text_sequence = []
    for day in all_categories:
        with open(
                f'IoT_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_day_{day}_{method}_th={threshold}_{model}_seq.pkl',
                'rb') as file:
            sequences = pickle.load(file)
        if sequences != None:
            text_sequence += sequences

    number_sequence = []
    for sublist in text_sequence:
        converted = []
        for index, text in enumerate(sublist):

            dict_index = index % len(dictionaries)
            current_dict = dictionaries[dict_index]
            flag = 0

            for t, num in current_dict.items():
                if t == text:
                    flag = 1
                    converted.append(num)
                elif dict_index == 3 and t == sublist[index - 1] + ':' + text:
                    flag = 1
                    converted.append(num)
            if flag == 0:
                converted.append(99999)

        number_sequence.append(converted)

    behavior_sequence = remove_quadruplets(number_sequence)

    print(behavior_sequence)
    print(len(behavior_sequence))

    with open(f'IoT_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.pkl',
              'wb') as f3:
        pickle.dump(behavior_sequence, f3)
    with open(f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq.pkl',
              'wb') as f3:
        pickle.dump(behavior_sequence, f3)


def Transnum_increase(dataset, new_env, threshold, method, model, all_categories, dictionaries):
    text_sequence = []
    for day in all_categories:
        with open(
                f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_increase.pkl',
                'rb') as file:
            sequences = pickle.load(file)
        if sequences != None:
            text_sequence += sequences

    number_sequence = []
    for sublist in text_sequence:
        converted = []
        for index, text in enumerate(sublist):

            dict_index = index % len(dictionaries)
            current_dict = dictionaries[dict_index]
            flag = 0

            for t, num in current_dict.items():
                if t == text:
                    flag = 1
                    converted.append(num)
                elif dict_index == 3 and t == sublist[index - 1] + ':' + text:
                    flag = 1
                    converted.append(num)
            if flag == 0:
                converted.append(99999)

        number_sequence.append(converted)

    behavior_sequence = remove_quadruplets(number_sequence)

    print(behavior_sequence)
    print(len(behavior_sequence))

    with open(
            f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_seq_increase.pkl',
            'wb') as f3:
        pickle.dump(behavior_sequence, f3)


def Transnum_filter(dataset, new_env, threshold, method, model, all_categories, dictionaries):
    text_sequence = []
    for day in all_categories:
        with open(
                f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_filter.pkl',
                'rb') as file:
            sequences = pickle.load(file)
        if sequences != None:
            text_sequence += sequences

    number_sequence = []
    for sublist in text_sequence:
        converted = []
        for index, text in enumerate(sublist):

            dict_index = index % len(dictionaries)
            current_dict = dictionaries[dict_index]
            flag = 0

            for t, num in current_dict.items():
                if t == text:
                    flag = 1
                    converted.append(num)
                elif dict_index == 3 and t == sublist[index - 1] + ':' + text:
                    flag = 1
                    converted.append(num)
            if flag == 0:
                converted.append(99999)

        number_sequence.append(converted)

    behavior_sequence = remove_quadruplets(number_sequence)

    print(behavior_sequence)
    print(len(behavior_sequence))

    with open(
            f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_seq_filter.pkl',
            'wb') as f3:
        pickle.dump(behavior_sequence, f3)
