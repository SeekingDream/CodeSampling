import re
from datasets import Dataset, load_dataset
import ast
from .codellm import CodeTask


class MyDataset(Dataset):
    def __init__(self, new_data: Dataset):

        super(MyDataset, self).__init__(new_data.data)

    @staticmethod
    def process_item(item):
        pass

    def __iter__(self):
        for item in self.data:
            yield CodeTask.from_dict(item)

    # def to_list(self) -> list:
    #     res = []
    #     for item in self.data:
    #         res.append(CodeTask.from_dict(item))
    #     return res


class HumanEvalData(MyDataset):
    def __init__(self):
        dataset = load_dataset("openai_humaneval", split='test')
        dataset = dataset.to_list()
        for d in dataset:
            with open(f"./dataset/{d['task_id']}.txt", 'r') as file:
                filtered_prompt = '\n'.join(file.readlines())
            d['prompt']=filtered_prompt
        dataset = Dataset.from_list(dataset)
        self.data_name = "HumanEval"

        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)

        super(HumanEvalData, self).__init__(new_data)

    def init_transform(self, item):
        import_st = ""
        test_cases = item["test"] + f'check({item["entry_point"]})'
        return CodeTask(
            dataset_name=self.data_name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prefix=item["prompt"],
            suffix="",
            solution=item["prompt"] + item["canonical_solution"],
            test_cases=item["test"],
            config={
                "entry_point": item["entry_point"],
                "import_st": import_st,
                "test_cases": test_cases
            }
        )




TAB = '    '

def split_at_last_function_signature(code: str):
    # Parse the code into an Abstract Syntax Tree (AST)
    tree = ast.parse(code)

    # Collect all function definitions
    function_defs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_defs.append(node)

    if not function_defs:
        raise ValueError("No function definitions found in the input code.")

    # Get the last function definition
    last_function = function_defs[-1]

    # Extract everything up to the last function signature (imports and previous functions)
    code_up_to_last_function = ''
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node != last_function:
            code_up_to_last_function += ast.unparse(node) + '\n'

    # Extract the signature of the last function definition (i.e., the 'def ...' part)
    function_signature = ast.unparse(last_function).split('\n')[0]  # Only take the first line

    # Extract the body of the last function
    function_body = '\n'.join(ast.unparse(last_function).split('\n')[1:])

    return code_up_to_last_function + function_signature, function_body

class MBPPData(MyDataset):
    def __init__(self):
        
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        self.data_name = "MBPP"
        new_data = []
        for d in dataset:
            new_data.append(self.init_transform(d).to_dict())
        new_data = Dataset.from_list(new_data)
        super(MBPPData, self).__init__(new_data)

    def init_transform(self, item):
        signature, body = split_at_last_function_signature(item["code"])
        prompt = signature + '\n' + f'{TAB}"""\n{TAB}{item["prompt"]}\n{TAB}"""\n'
        solution = prompt + body
        import_st = '\n'.join(item['test_imports'])
        solution = import_st + '\n' + solution
        test_cases = "\n".join(item['test_list'])

        return CodeTask(
            dataset_name=self.data_name,
            data_id=item["task_id"],
            lang="python",
            task_name="CG",
            prefix=prompt,
            suffix="",
            solution=solution,
            test_cases=item["test_list"],
            config={
                "import_st": import_st,
                "test_cases": test_cases
            }
        )

