import os
import torch
import time
import numpy as np
import regex
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src.methods import ExecutionDatabase

data_root = "/home/simin/Project/CodeSelection_hyx/coder_reviewer_reranking/samples"
data_split = "test"
tag = ""
model_list = [
    "codex001", "codex002",
    "codegen-2b", "codegen-6b",  "codex-cushman",
]
dataset_list = [
    "codet_humaneval",  "mbpp",  "mbpp_sanitized"
]
temperature = 0.4

save_dir = "exe_db"



def tuple2str(a):
    a_list = [str(kkk) for kkk in a]
    return '____'.join(a_list)


def process_trace(raw_trace):
    pattern = regex.compile("[a-zA-Z_][a-zA-Z0-9_]* = .*")
    trace = []
    lines = raw_trace[-1]
    if lines is None:
        return None
    for line in lines:
        # print(line)
        if 'var' in line and regex.search(pattern, line) is not None:
            # print(regex.search(pattern, line).xgroup())
            trace.append(regex.search(pattern, line).group())
    return trace


def get_parsed_trace(data):
    traces = {}
    for idx in data:
        post_trace_list = []
        for problem_id, item in enumerate(data[idx]):
            post_traces = [process_trace(raw_trace) for raw_trace in item['raw_trace']]
            post_trace_list.append(post_traces)
        traces[tuple2str(idx)] = post_trace_list
    return traces


def get_return_statement(raw_trace):
    lines = raw_trace[-1]
    if lines is None:
        return None
    for line in lines:
        # print(line)
        if 'return' in line:
            return line.split('return')[1].strip()
    return None


def get_return_var_name(data):
    problem_num = len(data[(0, 0)])
    all_return_val = [[] for _ in range(problem_num)]
    all_passed = [[] for _ in range(problem_num)]
    for idx in data:
        for problem_id, item in enumerate(data[idx]):
            post_trace = get_return_statement(item['raw_trace'][0])
            all_return_val[problem_id].append(post_trace)
            all_passed[problem_id].append(item["execution_result_full_pass"])

    passed_avg = np.array(all_passed).mean(axis=1)
    print()




for model in model_list:
    for dataset in dataset_list:
        task_name = f"{model}_{dataset}_{True}.tar"
        save_path = os.path.join(save_dir, task_name)
        data = torch.load(save_path)
        get_return_var_name(data.data)
        res = {}
        for idx in data.data:
            for prob_id, item in enumerate(data.data[idx]):
                if prob_id not in res:
                    res[prob_id] = []
                res[prob_id].append(item['execution_result_full_pass'])
        final_res = []
        for prob_id in res:
            final_res.append(np.mean(res[prob_id]))
        print(model, dataset, np.mean(final_res))
        # print(final_res)
