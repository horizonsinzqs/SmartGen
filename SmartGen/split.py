import pickle

import numpy as np

from dictionary import fr_actions_off, us_actions_off, sp_actions_off

day_index = 0
hour_index = 1
device_index = 2
action_index = 3

action_dic = {
    "fr": fr_actions_off.values(),
    "us": us_actions_off.values(),
    "sp": sp_actions_off.values()
}


def calculate_hours(day1, hour1, day2, hour2):
    total1 = day1 * 24 + hour1 * 3
    total2 = day2 * 24 + hour2 * 3
    if (day2 < day1) or (day2 == day1 and hour2 < hour1):
        total2 += 168
    return total2 - total1


def extract_interval(sequence):
    intervals = []
    intervals.append(0)
    for i in range(0, len(sequence[0]) - 1):
        day1, day2 = sequence[day_index][i], sequence[day_index][i + 1]
        hour1, hour2 = sequence[hour_index][i], sequence[hour_index][i + 1]
        intervals.append(calculate_hours(day1, hour1, day2, hour2))

    return intervals


def extract_total(sequence):
    intervals = []
    intervals.append(0)
    for i in range(0, len(sequence[0]) - 1):
        day1, day2 = sequence[day_index][i], sequence[day_index][i + 1]
        hour1, hour2 = sequence[hour_index][i], sequence[hour_index][i + 1]
        intervals.append(intervals[-1] + calculate_hours(day1, hour1, day2, hour2))

    return intervals


def semantic_judge(action, data_name):
    return action[-1] not in np.array(action_dic[data_name])


def split_list_by_value(interval_lst, data_lst, value, data_name):
    result = []
    sublist = []
    for (i, item) in enumerate(interval_lst):
        if item > value and semantic_judge(data_lst[:, i], data_name):
            if sublist:
                result.append(sublist)
            sublist = [data_lst[:, i]]
        else:
            sublist.append(data_lst[:, i])
    if sublist:
        result.append(sublist)
    return result


def split(file_path, interval_threshold, total_threshold, data_name):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    for (i, row) in enumerate(data):
        data[i] = np.reshape(np.array(row), (-1, 4)).T

    days = []
    hours = []
    devices = []
    actions = []
    length = []
    for (i, row) in enumerate(data):
        days.append(row[day_index])
        hours.append(row[hour_index])
        devices.append(row[device_index])
        actions.append(row[action_index])
        length.append(len(row[action_index]))

    data_split_by_interval = []
    for (i, row) in enumerate(data):
        if (len(row[0]) == 1):
            data_split_by_interval.append(row)
            continue
        else:
            intervals = extract_interval(row)
            res = split_list_by_value(intervals, data_lst=row, value=interval_threshold, data_name=data_name)
            for item in res:
                data_split_by_interval.append(np.array(item).T)

    data_split_by_total = []
    for (i, row) in enumerate(data_split_by_interval):
        if (len(row[0]) == 1):
            data_split_by_total.append(row.T)
            continue
        else:
            intervals = extract_total(row)
            res = split_list_by_value(intervals, data_lst=row, value=total_threshold, data_name=data_name)
            for item in res:
                data_split_by_total.append(np.array(item))

    for (i, row) in enumerate(data_split_by_total):
        data_split_by_total[i] = row.reshape(1, -1)
    result_list = [arr.tolist()[0] for arr in data_split_by_total]
    return result_list


def Split(dataset, ori_env, need_split):
    if need_split == 1:

        new_groups = split(file_path=f'IoT_data/{dataset}/{ori_env}/trn.pkl', interval_threshold=9, total_threshold=24,
                           data_name=dataset)

        with open(f'IoT_data/{dataset}/{ori_env}/split_trn.pkl', 'wb') as f3:
            pickle.dump(new_groups, f3)
    else:
        print('This is no split')
        with open(f'IoT_data/{dataset}/{ori_env}/trn.pkl', 'rb') as file:
            data = pickle.load(file)

        with open(f'IoT_data/{dataset}/{ori_env}/split_trn.pkl', 'wb') as f3:
            pickle.dump(data, f3)


def Split_test(dataset, new_env):
    new_groups = split(file_path=f'IoT_data/{dataset}/{new_env}/test.pkl', interval_threshold=9, total_threshold=24,
                       data_name=dataset)
    print(new_groups)
    print(len(new_groups))

    with open(f'IoT_data/{dataset}/{new_env}/split_test.pkl', 'wb') as f3:
        pickle.dump(new_groups, f3)


def Split_vld(dataset, new_env):
    new_groups = split(file_path=f'IoT_data/{dataset}/{new_env}/vld.pkl', interval_threshold=9, total_threshold=24,
                       data_name=dataset)
    print(new_groups)
    print(len(new_groups))

    with open(f'IoT_data/{dataset}/{new_env}/split_vld.pkl', 'wb') as f3:
        pickle.dump(new_groups, f3)
