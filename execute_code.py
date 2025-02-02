import types
import importlib.util
import multiprocessing

def execute_code_file(code_path, timeout=5):
    """
    Execute a Python file within a subprocess to prevent indefinite blocking.
    Args:
        code_path (str): Path to the Python file to be executed.
        timeout (int): Maximum execution time in seconds.

    Returns:
        dict: A dictionary containing the status (success or error) and details of the execution.
    """
    def run_script(result):
        try:
            spec = importlib.util.spec_from_file_location("loaded_script", code_path)
            loaded_script = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded_script)
            result["status"] = "success"
            result["message"] = ""
        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)

    manager = multiprocessing.Manager()
    result = manager.dict()
    
    process = multiprocessing.Process(target=run_script, args=(result,))
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "error", "message": f"Execution timed out after {timeout} seconds."}

    return dict(result)


# Example usage:
# Assuming 'example_script.py' contains valid Python code
# result = execute_code_file("example_script.py")
# print(result)


def execute_function_from_file(code_path, input_list):
    """
    Execute the Python function defined in a file on a list of inputs.

    Args:
        code_path (str): Path to the Python file containing the function.
        input_list (list): A list of inputs to be passed to the function.

    Returns:
        list: A list of outputs corresponding to each input.
    """
    # Load the module from the given file path
    spec = importlib.util.spec_from_file_location("loaded_module", code_path)
    loaded_module = importlib.util.module_from_spec(spec)

    try:
        # Execute the module
        spec.loader.exec_module(loaded_module)

        # Extract the function from the module
        function_name = next(
            (key for key, value in vars(loaded_module).items() if isinstance(value, types.FunctionType)),
            None
        )

        if not function_name:
            raise ValueError("No valid function found in the provided file.")

        # Retrieve the function
        func = getattr(loaded_module, function_name)

        # Execute the function for each input in the input list
        outputs = [func(*args) if isinstance(args, (list, tuple)) else func(args) for args in input_list]

        return outputs

    except Exception as e:
        raise RuntimeError(f"An error occurred while executing the file: {e}")

# Example usage:
# Suppose the file at 'example_function.py' contains:
# def square(x):
#     return x * x

# inputs = [2, 3, 4, 5]
# result = execute_function_from_file("example_function.py", inputs)
# print(result)  # Output: [4, 9, 16, 25]


def execute_code_with_inputs(code_str, input_list):
    """
    Execute the given Python function code string on a list of inputs.

    Args:
        code_str (str): A string containing a Python function definition.
        input_list (list): A list of inputs to be passed to the function.

    Returns:
        list: A list of outputs corresponding to each input.
    """
    # Define a dictionary to serve as the execution context for the code string
    exec_context = {}

    # Execute the code string to define the function in the context
    exec(code_str, exec_context)

    # Extract the function from the context
    function_name = next(
        (key for key, value in exec_context.items() if isinstance(value, types.FunctionType)),
        None
    )

    if not function_name:
        raise ValueError("No valid function found in the provided code string.")

    # Retrieve the function
    func = exec_context[function_name]

    # Execute the function for each input in the input list
    outputs = [func(*args) if isinstance(args, (list, tuple)) else func(args) for args in input_list]

    return outputs

# Example usage:
code = """
def square(x):
    return x * x
"""
inputs = [2, 3, 4, 5]
result = execute_code_with_inputs(code, inputs)
print(result)  # Output: [4, 9, 16, 25]

