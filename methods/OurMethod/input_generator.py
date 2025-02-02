
from typing import List

import subprocess
import ast
import os


def extract_func_name(code_str):
    """
    Extract the function name from a Python function code string.

    Args:
        code_str (str): The Python function code string.

    Returns:
        str: The function name.
    """
    try:
        parsed_ast = ast.parse(code_str)
        for node in parsed_ast.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
        return None
    except SyntaxError as e:
        print("SyntaxError:", str(e))
        return None


# class MySimpleConcolicFuzzer(SimpleConcolicFuzzer):
#     def __init__(self, init_val):
#         self.init_val = init_val
#         super(MySimpleConcolicFuzzer, self).__init__()
#

class InputGenerator:
    def __init__(self, bin_path: str, work_dir: str):
        self.bin_path = bin_path
        self.work_dir = work_dir

        # self.prog_candidates = prog_candidates
        # self.demo_inputs = demo_inputs
        # self.python = "/home/simin/miniconda3/envs/codetune/bin/python"
        # # self.python = "/home/simin/python_lib/Python-3.2.3/python"
        # self.PyExZ3 = "/home/simin/python_lib/PyExZ3/pyexz3.py"

    def code2func(self, code_string, globals_dict=None, locals_dict=None):
        """
        Transform a Python code string into a function pointer.

        Args:
            code_string (str): The Python code string to transform into a function.
            globals_dict (dict, optional): Global variables dictionary. Defaults to None.
            locals_dict (dict, optional): Local variables dictionary. Defaults to None.

        Returns:
            function: A function pointer representing the code in the code string.
        """
        if globals_dict is None:
            globals_dict = {}
        if locals_dict is None:
            locals_dict = {}

        try:
            exec(code_string, globals_dict, locals_dict)
            function_pointer = locals_dict.get(next(iter(locals_dict)), None)
            if callable(function_pointer):
                return function_pointer
            else:
                raise ValueError("The code string did not define a callable function.")
        except Exception as e:
            raise ValueError(f"Error while transforming code string into a function: {str(e)}")

    def string2file(self, string, file_name):
        with open(file_name, 'w') as f:
            f.writelines(string)

    def parse_output(self, output_str):
        output_list = eval(output_str.split('\n')[-2])
        generated_input = [d['input'] for d in output_list]
        return generated_input

    def generate_input(self, prog_code):
        func_name = extract_func_name(prog_code)
        prog_file = os.path.join(self.work_dir, f'tmp/{func_name}.py')

        # input_file = f"/tmp/{func_name}_input.py"
        # input_str = f"INI_ARGS = {init_val}"

        self.string2file(prog_code, prog_file)

        command = (f"{self.bin_path} "
                   f"cover tmp.{func_name}.{func_name} "
                   f"--coverage_type opcode "
                   f"--max_uninteresting_iterations 5"
                   )
        try:
            result = subprocess.run(
                command, shell=True, timeout=20,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.returncode, result.stdout
        except Exception as e:
            return -1, None
