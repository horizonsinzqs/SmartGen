import argparse
import pickle
import ast
import os
import json
from openai import OpenAI
from dictionary import dayofweek_dict, hour_dict, fr_devices_dict, fr_actions, sp_devices_dict, sp_actions, us_devices_dict, us_actions
from split import Split
from dayse import Dayse
from transtext import Transtext
from transnumber import Transnum
from sppc import SPPC_select, similarity_select
from baseline1 import Anomaly_detection
from baseline2 import Train
from text_translation_matrix import ATM
from extract import Extract
from find_categories import Find_categories
from security_check import security_check
from SAS_main import SASRec_behavior_prediction

vocab_dic = {"an": 141, "fr": 223, "us": 269, "sp": 235}
device_dic = {"us": us_devices_dict, "fr": fr_devices_dict, "sp": sp_devices_dict}
act_dic = {"us": us_actions, "fr": fr_actions, "sp": sp_actions}


def get_args_parser():
    parser = argparse.ArgumentParser('LLM generation', add_help=False)
    parser.add_argument('--model', default='gpt-4o', type=str,
                        help='The used LLM: Llama_405B/70B/gpt-4o/deepseek-v3')
    parser.add_argument('--dataset', default='us', type=str,
                        help='Name of dataset to train: an/fr/us/sp')
    parser.add_argument('--ori_env', default='winter', type=str,
                        help='The original home environment: winter/daytime')
    parser.add_argument('--new_env', default='spring', type=str,
                        help='The new home environment: spring/night')
    parser.add_argument('--method', default='SPPC', type=str,
                        help='The compression method: SPPC/similarity/instance')
    parser.add_argument('--threshold', default=0.905, type=float,
                        help="The compression threshold")
    parser.add_argument('--need_test', default=True, type=bool,
                        help='The experimental setup: True/False')
    parser.add_argument('--need_generate', default=False, type=bool,
                        help='The experimental setup: True/False')
    return parser


def LLM_call(openai_client, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]


    response = openai_client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        # model="Meta-Llama-3.3-70B-Instruct",
        # model="qwen2.5-72b-instruct",
        stream=False,
        messages=messages,
        max_tokens=8040,
        temperature=0,
        top_p=0,
        seed=2024
    )

    response = response.choices[0].message.content.strip()
    # response = response.choices[0].message
    print(response)
    # response = cleanup_response(response)

    return response

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print(args)

    if args.need_generate:
        Split(args.dataset, args.ori_env, 1)
        Dayse(args.dataset, args.ori_env)
        if args.method == 'SPPC':
            Train(args.dataset, args.ori_env, vocab_dic[args.dataset])
            SPPC_select(args.dataset, args.ori_env, vocab_dic[args.dataset], args.threshold)
        elif args.method == 'similarity':
            similarity_select(args.dataset, args.ori_env, args.threshold)

        all_categories = Find_categories(args.dataset, args.ori_env, args.method, args.threshold)
        device_dict = device_dic[args.dataset]
        actions = act_dic[args.dataset]
        dictionaries = [dayofweek_dict, hour_dict, device_dict, actions]

        ATM(args.dataset, args.ori_env, actions)
        Transtext(args.dataset, args.ori_env, args.threshold, args.method, all_categories, dictionaries)

        if args.new_env == 'spring':
            sentence = f'The previous environment is {args.ori_env}. The changed environment is warm {args.new_env}.'
        elif args.new_env == 'night':
            sentence = f'The previous environment: user is active during the {args.ori_env} and rest at {args.new_env}. The changed environment: user is active at {args.new_env} and rest during the {args.ori_env}.'
        elif args.new_env == 'multiple':
            sentence = f'The previous environment was for a {args.ori_env} person to be at home, and the changed environment is for {args.new_env} people to be at home'

        with open(f'{args.dataset}_keys_best.txt', 'r') as file:
            device_control_dict = file.read()

        with open(f'IoT_data/{args.dataset}/{args.ori_env}/action_transitions.json', 'r', encoding='utf-8') as f:
            action_transition = json.load(f)

        for day in all_categories:
            with open(f'IoT_data/{args.dataset}/{args.ori_env}/trn_day_{day}_{args.method}_th={args.threshold}_text.pkl', 'rb') as file3:
                user_sequence = pickle.load(file3)
                print(len(user_sequence))
            # gpt
            openai_client = OpenAI(
                api_key=" ",
                base_url=" ",
            )
            # Llama
            # openai_client = OpenAI(
            #     api_key=" ",
            #     base_url=" ",
            # )
            # Qwen
            # openai_client = OpenAI(
            #     api_key=" ",
            #     base_url=" ",
            # )
            prompt = "You're an IoT expert. And you are very knowledgeable about user behavior and habits in smart homes. Now, the user would like to ask you about the possible changes in user behavior sequence after the change of environment. " \
                     "The user will provide you with the user's previous life environment and the changed environment, the user's previous behavior sequence, and a set of devices and device states. And the user hope that you can use your knowledge and the set to generate possible user behavior sequences after the change based on the original sequences." \
                     "Each user behavior sequence consists of some quadruples containing the number of weeks, hours, devices." \
                     f"The set of the possible device and device states: {device_control_dict}" \
                     f"{sentence} The user's compressed original sequences of behavior: {user_sequence}. User's behavior habits: {action_transition}" \
                     "Your task: First, select the possible new device states from the set of devices and device states which are also possible new user behaviors. " \
                     "The second step is to reasonably add possible new user behaviors to the original user behavior sequences. The third step is to reasonably continue and expand the sequence based on user behavior habits." \
                     "Requirements:" \
                     "1.Please consider the devices that will be used in the new environment as widely as possible based on the set of devices." \
                     "2.Please strictly follow the correspondence between the devices and device states to generate. Do not generate device states that do not match the device." \
                     "3.Please add as many new devices and device behaviors as possible to better adapt to changes in the environment." \
                     "4.Please make sure that the generated sequence is not a single behavior, but a sequence of consecutive behaviors." \
                     "5.Please also generate reasonable behavior time when generating, not just a single behavior." \
                     "6.The final generated behavior sequences set is in the format of <seq [['...'], ['...'], ['...']] seq>. For example, the sequences set can be like <seq [['Sunday', '(21~24)', 'Blind', 'Blind:windowShade open', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement charging', 'Sunday', '(21~24)', 'Camera', 'Camera:notification', 'Sunday', '(21~24)', 'Blind', 'Blind:windowShade close', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Sunday', '(21~24)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning'], ['Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'RobotCleaner', 'RobotCleaner:setRobotCleanerMovement cleaning', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade open', 'Friday', '(0~3)', 'Camera', 'Camera:notification', 'Friday', '(0~3)', 'Blind', 'Blind:windowShade close']] seq>" \
                     "Note that each [...] subsequence represents the user's behavior over a period of time. There is no direct correlation between subsequences. At the same time, the final sequence is strictly generated in the format of <seq [['......'], ['......'], ['......']] seq> without line breaks or inconsistent formats." \
                     "Please think step by step, and return the final generated user behavior sequence set."
            response = LLM_call(openai_client, prompt)
            with open(f'IoT_data/{args.dataset}/{args.new_env}/{args.dataset}_{args.new_env}_generation_day_{day}_{args.method}_th={args.threshold}_{args.model}.pkl', 'wb') as f3:
                pickle.dump(response, f3)

        Extract(args.dataset, args.new_env, args.threshold, args.method, args.model, all_categories)
        Transnum(args.dataset, args.new_env, args.threshold, args.method, args.model, all_categories, dictionaries)
        security_check(args.dataset, args.new_env, args.threshold, args.method, args.model)


    if args.need_test:
        Anomaly_detection(args.dataset, args.new_env, args.threshold, args.method, args.model)
        # SASRec_behavior_prediction(args.dataset, args.new_env, args.threshold, args.method, args.model, need='train')
        # SASRec_behavior_prediction(args.dataset, args.new_env, args.threshold, args.method, args.model, need='test')

















