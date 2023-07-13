# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse

from utils import get_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mbpp_sanitized", "codet_humaneval", "humaneval"],
    )
    parser.add_argument("--num_seeds", type=int, default=25)
    parser.add_argument("--num_samples", type=int, default=5)

    args = parser.parse_args()

    dataset = get_dataset(args.dataset)



    args.output_path = args.output_path + "/" + str(args.dataset)
    if args.dataset == "codet_humaneval":
        data_file_path = "dataset/human_eval/dataset/CodeTHumanEval.jsonl"
    elif args.dataset == "mbpp_sanitized":
        data_file_path = "dataset/mbpp/mbpp_sanitized_for_code_generation.jsonl"
    args.end_template = ["\nclass", "\ndef", "\n#", "\nif"]
    collector = CollectorWithInfo.from_args(args, dataset)
    if args.temperature > 0:
        args.seed = list(range(args.num_seeds))
        for i in range(args.num_samples):
            collector(i, i, 5)
    else:
        args.seed = list(range(args.num_seeds))
        collector(0, 0, 1)
