import json
import pickle
from typing import List, Tuple, Dict

import numpy as np


class LinkAnalyzer:
    def __init__(self, actions: Dict[str, int]):
        self.actions = actions
        self.number_set = set()
        self.transition_matrix = None
        self.number_to_index = {}
        self.index_to_number = {}
        self.link_counts = {}

        for action, index in actions.items():
            self.index_to_number[index] = action

    def fit_sequences(self, sequences: List[List[int]]):

        for numbers in sequences:
            self.number_set.update(numbers)

        for idx, num in enumerate(sorted(self.number_set)):
            self.number_to_index[num] = idx
            self.link_counts[num] = {}

        for numbers in sequences:
            for i in range(len(numbers) - 1):
                current_num = numbers[i]
                next_num = numbers[i + 1]

                if next_num not in self.link_counts[current_num]:
                    self.link_counts[current_num][next_num] = 0
                self.link_counts[current_num][next_num] += 1

    def get_top_transitions(self, top_n: int = 5) -> Dict[int, List[Tuple[int, int]]]:
        top_transitions = {}
        for num, transitions in self.link_counts.items():
            sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:top_n]
            top_transitions[num] = sorted_transitions
        return top_transitions

    def print_transition_matrix(self):
        num_count = len(self.number_set)
        self.transition_matrix = np.zeros((num_count, num_count), dtype=int)

        for current_num, transitions in self.link_counts.items():
            current_idx = self.number_to_index[current_num]
            for next_num, count in transitions.items():
                next_idx = self.number_to_index[next_num]
                self.transition_matrix[current_idx][next_idx] = count

        print("Link matrix (rows represent current action, list represents next action): ")
        print("  ", end="")
        for num in sorted(self.number_set):
            print(f"{self.index_to_number[num]:>30}", end=" ")
        print()

        for i, num in enumerate(sorted(self.number_set)):
            print(f"{self.index_to_number[num]:>30} ", end="")
            for j in range(num_count):
                print(f"{self.transition_matrix[i][j]:>3}", end=" ")
            print()


def analyze_link(sequences: List[List[int]], actions: Dict[str, int], file_name):
    analyzer = LinkAnalyzer(actions)
    analyzer.fit_sequences(sequences)

    top_transitions = analyzer.get_top_transitions(top_n=5)

    print("The top 5 most commonly followed actions after each action: ")
    for num, transitions in top_transitions.items():
        if not transitions:
            print(f"Actions {analyzer.index_to_number[num]} are usually not followed by any other actions")
        else:
            print(f"Actions {analyzer.index_to_number[num]} most common action afterwards: ")
            for next_num, count in transitions:
                print(f"  -> {analyzer.index_to_number[next_num]}: {count} times")

    transition_results = {}

    for num, transitions in top_transitions.items():
        action_key = str(analyzer.index_to_number[num])

        if not transitions:
            transition_results[action_key] = {
                "message": "It is usually not followed by the next action.",

            }
        else:
            transition_data = []
            for next_num, count in transitions:
                transition_data.append({
                    "next_action": analyzer.index_to_number[next_num],
                    "count": count
                })

            transition_results[action_key] = {
                "message": "The most common action following it",
                "transitions": transition_data
            }

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(transition_results, f, ensure_ascii=False, indent=4)

    analyzer.print_transition_matrix()


def ATM(dataset, ori_env, actions):
    with open(f'IoT_data/{dataset}/{ori_env}/split_trn.pkl', 'rb') as file2:
        data = pickle.load(file2)

    indices_to_extract = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]

    sequences = [
        [sublist[i] for i in indices_to_extract if i < len(sublist)]
        for sublist in data
    ]
    file_name = f'IoT_data/{dataset}/{ori_env}/action_transitions.json'
    analyze_link(sequences, actions, file_name)

    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(json.dumps(data, ensure_ascii=False, indent=2))
