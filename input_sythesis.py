import os
import json


import torch
from tqdm import tqdm
from argparse import ArgumentParser

from src.methods.OurMethod import InputGenerator
from utils import DATA_LIST, MODEL_LIST, SHOT_DICT, SPLIT_TAG

raw_code_dir = f"{os.environ['HOME']}/Project/CodeSampling/samples"
save_dir = f"{os.environ['HOME']}/Project/CodeSampling/SyntheticInput"
tmp_dir = f"{os.environ['HOME']}/Project/CodeSampling/tmp"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
if not os.path.isdir(tmp_dir):
    os.mkdir(tmp_dir)

def construct_code(sample, model_id, data_id):
    if data_id == 0:
        return sample['src'] + sample['trg_prediction']
    elif data_id == 1:
        if model_id in [3, 4]:
            return sample['src'] + sample['trg_prediction']
        else:
            return sample['trg_prediction']
    elif data_id == 2:
        return sample['src'] + sample['trg_prediction']
    else:
        raise NotImplementedError


def load_code_candidates(model_id, data_id):
    model = MODEL_LIST[model_id]
    data = DATA_LIST[data_id]
    task_dir = os.path.join(raw_code_dir, os.path.join(model, data))
    task_name = model + SPLIT_TAG + data
    print("TASK NAME", task_name)
    shot_num = SHOT_DICT[task_name]
    assert len(os.listdir(task_dir)) == 25

    all_generated_codes = {}
    for seed_name in os.listdir(task_dir):
        work_dir = os.path.join(task_dir, seed_name)
        work_dir = os.path.join(work_dir, os.path.join(f"{shot_num}-shot", "sample-0.4"))
        for num in range(5):
            file_name = os.path.join(work_dir, f"test-{num}.jsonl")

            all_data = []
            with open(file_name, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    all_data.append(data)
            one_trial_codes = [construct_code(d, model_id, data_id) for d in all_data]
            trial_key = seed_name + SPLIT_TAG + str(num)
            all_generated_codes[trial_key] = one_trial_codes

    picked_keys = sorted(all_generated_codes.keys())[:60]
    picked_codes = {
        picked_key: all_generated_codes[picked_key]
        for picked_key in picked_keys
    }
    return picked_codes


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=int, default=4)
    parser.add_argument("--data_id", type=int, default=0)
    args = parser.parse_args()
    return args

def main(args):
    model_id, data_id = args.model_id, args.data_id
    save_path = os.path.join(save_dir, f"{model_id}{SPLIT_TAG}{data_id}.tar")
    print(save_path)
    if not os.path.isfile(save_path):
        all_inputs = {}
    else:
        all_inputs = torch.load(save_path, weights_only=False)

    code_candidates = load_code_candidates(model_id, data_id)
    k = InputGenerator(
        bin_path=f"{os.environ['HOME']}/crosshair",
        work_dir=f"{os.environ['HOME']}/Project/CodeSampling"
    )

    for trial_id in code_candidates:
        if trial_id in all_inputs:
            print(model_id, data_id, trial_id, 'pass')
            continue
        code_list = code_candidates[trial_id]
        generated_inputs_list = []
        for i, code in tqdm(enumerate(code_list)):
            generated_inputs = k.generate_input(code)
            print(generated_inputs)
            print()
            generated_inputs_list.append(generated_inputs)
        all_inputs[trial_id] = generated_inputs_list
        torch.save(all_inputs, save_path)
        print(model_id, data_id, trial_id, 'successful')


if __name__ == '__main__':
    args = parse_args()
    main(args)

