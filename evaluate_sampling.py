from execute_code import execute_code_with_inputs, execute_function_from_file, execute_code_file
from utils import *
import argparse
import tempfile
import jsonlines
import ast, astor
def extract_assert_statements(code_snippet):
    tree = ast.parse(code_snippet)
    assert_statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            assert_source = astor.to_source(node)
            if assert_source.find("assert") != -1:
                assert_statements.append(assert_source)
    return assert_statements

def evaluate(dataset, sample_num, sample_dir, save_dir):
    dataset_list = dataset.to_list()
    dataset = [CodeTask.from_dict(d) for d in dataset_list]

    # for i in range(len(data)):
    #     task_dir = os.path.join(save_dir, str(i))
        

    # final_dataset = []
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir, exist_ok=True)

    for data in dataset:

        file_id = str(data.data_id).replace('/', '-')
        task_dir = os.path.join(sample_dir, file_id)

        std_test_cases = extract_assert_statements(data.test_cases)
        head = data.test_cases[:data.test_cases.find('assert')]
        sample_execution_results = []
        all_sample_execution_results = []

        for i in range(sample_num):
            for single_test_case in std_test_cases:
                sample_path = os.path.join(task_dir, f"sample_{i}.py")
                with open(sample_path, 'r') as r:
                    code = r.read()
                code += ('\n\n\n'+head+single_test_case)
                with tempfile.TemporaryDirectory() as temp_dir:                
                    temp_file_path = os.path.join(temp_dir, "test_sample_{i}.py")
                    with open(temp_file_path, "w") as f:
                        f.write(code)
                    result = execute_code_file(temp_file_path)
                    sample_execution_results.append(result)
            all_sample_execution_results.append(sample_execution_results)
        # print(save_dir)
        with jsonlines.open(save_dir, 'a') as writer:
            writer.write({
                'data_id': data.data_id,
                'all_exe_results': all_sample_execution_results
            })
    # pass

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, default=0)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--sample_dir', type=str, default='')
    parser.add_argument('--model_id', type=str, default='')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--eval_dir', type=str, required=True)
    parser.add_argument('--overwrite', type=str, default=False)

    args = parser.parse_args()

    dataset = load_dataset(args.data_id)
    data_name = "HumanEval" if args.data_id==0 else "MBPP_sanitized"
    
    sample_num = args.n
    save_dir = f"{args.eval_dir}/modelID-{args.model_id}/temperature-{args.temperature}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_dir += f"{data_name}.jsonl"
    sample_dir = f"{args.sample_dir}/modelID-{args.model_id}/{data_name}/temperature-{args.temperature}"

    if os.path.exists(save_dir):
        if not args.overwrite:
            print(f"Evaluation already exists in \"{save_dir}\".")
        else:
            os.remove(save_dir)
    if not os.path.exists(save_dir):
        evaluate(dataset, sample_num, sample_dir, save_dir)