import pickle


def Transtext(dataset, ori_env, threshold, method, all_categories, dictionaries):
    for day in all_categories:

        with open(f"IoT_data/{dataset}/{ori_env}/trn_day_{day}_{method}_th={threshold}.pkl", 'rb') as file3:
            X = pickle.load(file3)
        number_sequence = X

        text_sequence = []
        for sublist in number_sequence:
            converted = []
            for index, number in enumerate(sublist):
                dict_index = index % len(dictionaries)
                current_dict = dictionaries[dict_index]

                text = next((text for text, num in current_dict.items() if num == number), str(number))
                converted.append(text)
            text_sequence.append(converted)

        print(text_sequence)
        print(len(text_sequence))

        with open(f"IoT_data/{dataset}/{ori_env}/trn_day_{day}_{method}_th={threshold}_text.pkl", 'wb') as f3:
            pickle.dump(text_sequence, f3)


def Transtext_over(dictionaries):
    with open(f"data/fr_data/deleted_flattened_useful_fr_trn_instance_10.pkl", 'rb') as file3:
        X = pickle.load(file3)
    number_sequence = X

    text_sequence = []
    for sublist in number_sequence:
        converted = []
        for index, number in enumerate(sublist):
            dict_index = index % len(dictionaries)
            current_dict = dictionaries[dict_index]

            text = next((text for text, num in current_dict.items() if num == number), str(number))
            converted.append(text)
        text_sequence.append(converted)

    print(text_sequence)
    print(len(text_sequence))

    with open(f"fr_trn_text_2.pkl", 'wb') as f3:
        pickle.dump(text_sequence, f3)


def Transtext_increase(dataset, new_env, threshold, method, model, all_categories, dictionaries):
    for day in all_categories:
        with open(
                f"increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}.pkl",
                'rb') as file3:
            X = pickle.load(file3)
        number_sequence = X

        text_sequence = []
        for sublist in number_sequence:
            converted = []
            for index, number in enumerate(sublist):
                dict_index = index % len(dictionaries)
                current_dict = dictionaries[dict_index]

                text = next((text for text, num in current_dict.items() if num == number), str(number))
                converted.append(text)
            text_sequence.append(converted)

        print(text_sequence)
        print(len(text_sequence))

        with open(
                f"increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_text.pkl",
                'wb') as f3:
            pickle.dump(text_sequence, f3)


def Transtext_filter(dataset, new_env, threshold, method, model, all_categories, dictionaries):
    for day in all_categories:
        with open(
                f"filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}.pkl",
                'rb') as file3:
            X = pickle.load(file3)
        number_sequence = X

        text_sequence = []
        for sublist in number_sequence:
            converted = []
            for index, number in enumerate(sublist):
                dict_index = index % len(dictionaries)
                current_dict = dictionaries[dict_index]

                text = next((text for text, num in current_dict.items() if num == number), str(number))
                converted.append(text)
            text_sequence.append(converted)

        print(text_sequence)
        print(len(text_sequence))

        with open(
                f"filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_text.pkl",
                'wb') as f3:
            pickle.dump(text_sequence, f3)
