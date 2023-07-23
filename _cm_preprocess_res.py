import os
import torch
import time

from src.methods import ExecutionDatabase

data_root = "/home/simin/Project/CodeSelection_hyx/coder_reviewer_reranking/samples"
data_split = "test"
tag = ""
model_list = [
    "codegen-2b", "codegen-6b",  "codex-cushman",
    "codex001", "codex002"
]
dataset_list = [
    "codet_humaneval",  "mbpp",  "mbpp_sanitized"
]
temperature = 0.4

rejection_opt = [False, True]
save_dir = "exe_db"
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

for is_no_rejection in rejection_opt:
    for model in model_list:
        for dataset in dataset_list:
            data_path = f"{data_root}/{model}/{dataset}"
            data_path = f"{data_path}/seed-*/0-shot/*-{temperature}"
            task_name = f"{model}_{dataset}_{is_no_rejection}.tar"
            save_path = os.path.join(save_dir, task_name)

            t1 = time.time()
            exe_res = ExecutionDatabase(
                data_path,
                data_split,
                "humaneval",
                tag=tag,
                model=model,
                verbose=False,
                dataset=dataset,
                no_rejection=is_no_rejection,
            )
            torch.save(exe_res, save_path)
            t2 = time.time()
            print(task_name, "success", t2 - t1)


