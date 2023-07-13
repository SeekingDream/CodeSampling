# Copyright (c) Meta Platforms, Inc. and affiliates.

import bashlex
import collections
import json
import pickle
import numpy as np
import os
import random
from glob import glob
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm, trange
import torch
from argparse import ArgumentParser
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from datasets import load_metric

from src.methods import ExecutionDatabase
from src.methods import OracleSelector
from src.methods import LogProbSelector
from src.methods import RandomSelector
from src.methods import LogProbEnsembleSelector
from src.methods import ExeLogProbSelector
from src.methods import ExeRandomSelector
from src.methods import ExeLogProbEnsembleSelector



import fasttext
from huggingface_hub import hf_hub_download

STATIC_LOGPROPB_METHODS = [
    "sum_logprob", "avg_logprob",
    "avg_reverse_logprob", "sum_reverse_logprob",
]
EXE_LOGPROPB_METHODS = [
    "executability-sum_logprob",
    "executability-avg_logprob",
    "executability-avg_reverse_logprob",
    "executability-sum_reverse_logprob",
]
OURMETHOD = ["our_trace_based"]
MBR = ["mbr_exec"]

METHOD_LIST = STATIC_LOGPROPB_METHODS + EXE_LOGPROPB_METHODS + [
    "oracle", "random",

    "avgreverselogprob-ensemble#0.5",
    "sumreverselogprob-ensemble#0.5",

    "executability-random",
    "executability-avgreverselogprob-ensemble#0.5",
    "executability-sumreverselogprob-ensemble#0.5",
]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="./result_db")
    parser.add_argument(
        "--model",
        default="codegen-2b",
        type=str,
        choices=[
            "codegen-2b",
            "codegen-2B-half",
            "codegen-6B",
            "codegen-6B-half",
            "codegen-16B-half",
            "incoder-1B",
            "incoder-1B-half",
            "incoder-6B",
            "incoder-6B-half",
            "codex001",
            "codex002",
            "codex-cushman",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="codet_humaneval",
        choices=[
            "mbpp",
            "spider",
            "nl2bash",
            "humaneval",
            "codet_humaneval",
            "mbpp_sanitized",
            "mbpp_hyx",
        ],
    )
    parser.add_argument("--num_samples_start", type=int, default=24)
    parser.add_argument("--num_samples_end", type=int, default=25)
    parser.add_argument("--num_samples_gap", type=int, default=1)
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--num_bootstraps", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid_search_alpha", action="store_true", default=False)
    parser.add_argument("--ablate", action="store_true", default=False)
    parser.add_argument("--no-rejection", action="store_true", default=False)
    parser.add_argument(
        "--use_generated_assertions", action="store_true", default=False
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/simin/Project/CodeSelection_hyx/coder_reviewer_reranking/samples",
    )
    args = parser.parse_args()

    if args.temperature > 0:
        base_crits = ["sum_logprob", "avg_logprob", "avg_reverse_logprob", "random"]
        if args.grid_search_alpha:
            for i in range(1, 10):
                base_crits.append(
                    f"sumreverselogprob-ensemble#0.{i}",
                )
            for i in range(1, 10):
                base_crits.append(
                    f"avgreverselogprob-ensemble#0.{i}",
                )
        else:
            base_crits.append(
                f"sumreverselogprob-ensemble#0.5",
            )
            base_crits.append(
                f"avgreverselogprob-ensemble#0.5",
            )
        all_crits = base_crits + ["executability-" + b for b in base_crits]
    else:
        all_crits = ["sum_logprob"]

    args.criteria = METHOD_LIST # all_crits
    args.data_path = f"{args.data_path}/{args.model}/{args.dataset}"

    return args


def evaluate_selection(selected_dataset):
    all_passed = [d["execution_result_full_pass"] for d in selected_dataset]
    return sum(all_passed) / len(all_passed)


def compute_acc(args):
    humaneval_good_execution_result = 0
    data_path = f"{args.data_path}/seed-*/0-shot/*-{args.temperature}"
    if args.top_p != 1.0:
        data_path += f"-p{args.top_p}"
    if args.max_tokens != 512:
        data_path += f"-max{args.max_tokens}"

    exe_res = ExecutionDatabase(
        data_path,
        args.data_split,
        "humaneval",
        tag=args.tag,
        model=args.model,
        verbose=args.verbose,
        dataset=args.dataset,
        no_rejection=args.no_rejection,
    )

    id_keys = list(exe_res.data.keys())

    acc_dict = collections.defaultdict(list)
    std_dict = collections.defaultdict(list)

    for crit in args.criteria:
        print(crit)
        selector = get_selector(
            crit,
            exe_res,
            humaneval_good_execution_result,
            use_multi_assertions=args.use_generated_assertions,
        )

        step_results = collections.defaultdict(list)
        for sub_n_samples in trange(
                args.num_samples_start, args.num_samples_end, args.num_samples_gap
        ):
            random.seed(args.seed)
            np.random.seed(args.seed)
            for bootstrap_i in range(args.num_bootstraps):
                ids = random.sample(id_keys, sub_n_samples)
                # id_is = np.random.choice(list(range(len(id_keys))), size=sub_n_samples, replace=True)
                # ids = [id_keys[i] for i in id_is]

                selected = selector.select(ids)

                result = evaluate_selection(selected)
                step_results[sub_n_samples].append(result)
        for k, v in step_results.items():
            acc_dict[crit].append(np.mean(v))
            std_dict[crit].append(np.std(v))
    return (acc_dict, std_dict)


def get_selector(
        criterion,
        exe_res: ExecutionDatabase,
        mbpp_good_execution_result,
        use_multi_assertions=False
):
    '''
    这个函数的主要目的是根据给定的criterion条件选择样本，并返回一个用于样本选择的函数以及可选的用于排序的辅助函数。
    具体选择样本的逻辑和排序方式取决于criterion的不同取值。
    '''
    if "$" in criterion:
        criterion = criterion.split("$")[0]
    secondary_key_function = None
    if not use_multi_assertions:
        exe_func = exe_selection_function
    else:
        exe_func = multi_exe_selection_function

    if criterion == "oracle":
        selector = OracleSelector(exe_res)

    elif criterion == "mbr_exec":
        selector = None

    elif criterion == "our_trace_based":
        selector = None
        # sample_selection_function = our_function
        # secondary_key_function = our_function1

    elif criterion in STATIC_LOGPROPB_METHODS:
        selector = LogProbSelector(exe_res, criterion)

    elif criterion == "random":
        selector = RandomSelector(exe_res)

    elif criterion.startswith("avgreverselogprob-ensemble#"):
        selector = LogProbEnsembleSelector(exe_res, criterion)

    elif criterion.startswith("sumreverselogprob-ensemble#"):
        selector = LogProbEnsembleSelector(exe_res, criterion)

    elif criterion in EXE_LOGPROPB_METHODS:
        selector = ExeLogProbSelector(exe_res, criterion, exe_func, mbpp_good_execution_result)

    elif criterion == "executability-random":
        selector = ExeRandomSelector(exe_res, exe_func, mbpp_good_execution_result)

    elif criterion.startswith("executability-avgreverselogprob-ensemble#"):
        selector = ExeLogProbEnsembleSelector(exe_res, criterion, exe_func, mbpp_good_execution_result)

    elif criterion.startswith("executability-sumreverselogprob-ensemble#"):
        selector = ExeLogProbEnsembleSelector(exe_res, criterion, exe_func, mbpp_good_execution_result)

    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return selector


def exe_selection_function(x, good_execution_result=0):
    exec_res = x["execution_result"]
    return exec_res[0] == good_execution_result


def multi_exe_selection_function(x, good_execution_result=0):
    exec_res = x["gen_execution_result_full"]
    return sum([e[0] == good_execution_result for e in exec_res])


def main(args):
    args.data_split = "test"
    acc_dict, std_dict = compute_acc(args)

    if args.tag != "":
        out_path = Path(
            f"{args.out_dir}/{args.dataset}-{args.model}-temp{args.temperature}-{args.tag}"
        )
    else:
        out_path = Path(
            f"{args.out_dir}/{args.dataset}-{args.model}-temp{args.temperature}"
        )
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(acc_dict, out_path / "acc.pt")
    torch.save(std_dict, out_path / "std.pt")
    print(f"saving to {out_path}")
    for crit in args.criteria:
        print(crit, f"{acc_dict[crit][-1]:.4f} {std_dict[crit][-1]:.2f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
