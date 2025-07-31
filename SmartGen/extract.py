import ast
import pickle
import re


def Extract(dataset, new_env, threshold, method, model, all_categories):
    for day in all_categories:
        with open(
                f'IoT_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_day_{day}_{method}_th={threshold}_{model}.pkl',
                'rb') as file:
            text = pickle.load(file)

        pattern = r"(\[\[.*?\]\])"

        longest_match = None
        max_length = 0

        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1)
            content_length = len(content)

            if content_length > max_length:
                max_length = content_length
                longest_match = content

        if longest_match == None:
            print("Not Found")
            seq = None
            with open(
                    f'IoT_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_day_{day}_{method}_th={threshold}_{model}_seq.pkl',
                    'wb') as f3:
                pickle.dump(seq, f3)
        else:
            extracted_content = longest_match
            try:
                text_sequence = ast.literal_eval(extracted_content)
                print("解析成功:", text_sequence)
                with open(
                        f'IoT_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_day_{day}_{method}_th={threshold}_{model}_seq.pkl',
                        'wb') as f3:
                    pickle.dump(text_sequence, f3)
            except (ValueError, SyntaxError) as e:
                print("解析失败:", e)
                seq = None
                with open(
                        f'IoT_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_day_{day}_{method}_th={threshold}_{model}_seq.pkl',
                        'wb') as f3:
                    pickle.dump(seq, f3)


def Extract_increase(dataset, new_env, threshold, method, model, all_categories):
    for day in all_categories:
        with open(
                f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_text_increase.pkl',
                'rb') as file:
            text = pickle.load(file)

        pattern = r"(\[\[.*?\]\])"

        longest_match = None
        max_length = 0

        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1)
            content_length = len(content)

            if content_length > max_length:
                max_length = content_length
                longest_match = content

        if longest_match == None:
            print("Not Found")
            seq = None
            with open(
                    f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_increase.pkl',
                    'wb') as f3:
                pickle.dump(seq, f3)
        else:
            extracted_content = longest_match
            try:
                text_sequence = ast.literal_eval(extracted_content)
                print("解析成功:", text_sequence)
                with open(
                        f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_increase.pkl',
                        'wb') as f3:
                    pickle.dump(text_sequence, f3)
            except (ValueError, SyntaxError) as e:
                print("解析失败:", e)
                seq = None
                with open(
                        f'increase_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_increase.pkl',
                        'wb') as f3:
                    pickle.dump(seq, f3)


def Extract_filter(dataset, new_env, threshold, method, model, all_categories):
    for day in all_categories:
        with open(
                f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_text_filter.pkl',
                'rb') as file:
            text = pickle.load(file)

        pattern = r"(\[\[.*?\]\])"

        longest_match = None
        max_length = 0

        for match in re.finditer(pattern, text, re.DOTALL):
            content = match.group(1)
            content_length = len(content)

            if content_length > max_length:
                max_length = content_length
                longest_match = content

        if longest_match == None:
            print("Not Found")
            seq = None
            with open(
                    f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_filter.pkl',
                    'wb') as f3:
                pickle.dump(seq, f3)
        else:
            extracted_content = longest_match
            try:
                text_sequence = ast.literal_eval(extracted_content)
                print("解析成功:", text_sequence)
                with open(
                        f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_filter.pkl',
                        'wb') as f3:
                    pickle.dump(text_sequence, f3)
            except (ValueError, SyntaxError) as e:
                print("解析失败:", e)
                seq = None
                with open(
                        f'filter_data/{dataset}/{new_env}/{dataset}_{new_env}_generation_{method}_th={threshold}_{model}_seq_day_{day}_seq_filter.pkl',
                        'wb') as f3:
                    pickle.dump(seq, f3)
