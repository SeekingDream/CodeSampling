from utils import *
import argparse
from copy import deepcopy
import os
import torch
import jsonlines


def generate(code_llm, dataset, sample_num, save_dir, overwrite=False):
    dataset_list = dataset.to_list()
    dataset = [CodeTask.from_dict(d) for d in dataset_list]

    # for i in range(len(data)):
    #     task_dir = os.path.join(save_dir, str(i))
        

    final_dataset = []
    for data in dataset:
        for i in range(sample_num):
            tmp_data = deepcopy(data)
            file_id = str(tmp_data.data_id).replace('/', '-')
            task_dir = os.path.join(save_dir, file_id)
            if not os.path.exists(task_dir):
                os.makedirs(task_dir, exist_ok=True)
            save_path = os.path.join(task_dir, f"sample_{i}.py")

            tmp_data.save_path = save_path
            tmp_data.task_dir = task_dir
            final_dataset.append(tmp_data)
    res = code_llm.code_gen_batch(final_dataset)
    raw_res_path = os.path.join(save_dir, 'raw_res.tar')
    if overwrite or not os.path.exists(raw_res_path):
        torch.save(res, raw_res_path)
    for (output, logits) in res:
        save_file = output.original_task.save_path

        import_st = output.original_task.config["import_st"]
        # test_cases = output.original_task.config["test_cases"]
        final_code = import_st + '\n\n\n' + output.final_code #+ '\n\n\n' + test_cases

        if overwrite or not os.path.exists(save_file):
            with open(save_file, 'w') as f:
                f.write(final_code)
    # for lo in logits:
        logit_file_path = os.path.join(output.original_task.task_dir, f"logits.jsonl")
        if overwrite or not os.path.exists(logit_file_path):
            with jsonlines.open(logit_file_path, 'a') as writer:
                writer.write({
                    'sample_file': output.original_task.save_path.split('/')[-1],
                    'logits': logits
                })
    pass

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--model_id', type=int, required=True)
    parser.add_argument('--data_id', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default="./results/generated_code")
    parser.add_argument('--overwrite', type=bool, default=False)

    args = parser.parse_args()

    sample_num = args.n
    if args.temperature == 0:
        sample_num = 1
    dataset = load_dataset(args.data_id)
    code_llm = load_model(args.model_id)

    config = {
        'temperature':  args.temperature, # 0.6/0.8/0
        "top_p": args.top_p,# 0.95
        "max_tokens": 1024,  # args.max_tokens,
        "tp_size": 1,  # args.tp_size,
        "dtype": "float16",
        'lora_path': None,
        "stop": [
            "\n>>>", "\n$", '\nclass',
            '\ndef', '\n#', '\nprint',
            "\n@", "\nif __name__ == '__main__':"
        ]
    }
    code_llm.init_ai_kwargs(config)

    dataset_name = "HumanEval" if args.data_id==0 else "MBPP_sanitized"
    save_path = f"{args.save_dir}/modelID-{args.model_id}/{dataset_name}/temperature-{args.temperature}/"
    generate(code_llm, dataset, sample_num, save_path, args.overwrite)