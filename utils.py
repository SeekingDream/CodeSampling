import os


from data import MBPPGoogleDataset, HumanEvalDataset, MBPPSanDataset


WORKDIR = './working_dir'
if not os.path.isdir(WORKDIR):
    os.mkdir(WORKDIR)
RAW_CODE_DIR = './working_dir/raw_code'
if not os.path.isdir(WORKDIR):
    os.mkdir(WORKDIR)
ORDERED_CODE_DIR = './working_dir/ordered_code'
if not os.path.isdir(ORDERED_CODE_DIR):
    os.mkdir(ORDERED_CODE_DIR)


def get_dataset(data_name):
    if data_name == 'humaneval':
        data_file_path = "dataset/human_eval/dataset/CodeTHumanEval.jsonl"
        data_module = HumanEvalDataset(path=data_file_path, mode="prompt_only")
        _, dataset = data_module.extract_data()
    elif data_name == "mbpp_santized":
        data_file_path = "dataset/mbpp/mbpp_sanitized_for_code_generation.jsonl"
        data_module = MBPPSanDataset(path=data_file_path, mode="prompt_only")
        _, dataset = data_module.extract_data()
    else:
        raise ValueError('Invalid param')
    return dataset


if __name__ == '__main__':
    for data_name in ["mbpp_santized", "humaneval"]:
        dataset = get_dataset(data_name)
        print(len(dataset))