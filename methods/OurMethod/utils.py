import ast
import astor
from typing import List


import builtins


def is_builtin_function(func_name):
    """
    Check if a Python function is a built-in function.

    Args:
        func_name: A Python function name.

    Returns:
        True if the function is a built-in function, False otherwise.
    """
    return hasattr(builtins, func_name)


def extract_assert_statements(code_snippet):
    tree = ast.parse(code_snippet)
    assert_statements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            assert_source = astor.to_source(node)
            if assert_source.find("assert") != -1:
                assert_statements.append(assert_source)
    return assert_statements


def parse_args(left_node, func_entry):
    for node in ast.walk(left_node):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id'):
                if node.func.id == func_entry:
                    return node.args
                if not is_builtin_function(node.func.id):
                    return node.args
    return None


def parse_func(left_node):
    all_func = []
    for node in ast.walk(left_node):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'id'):
                all_func.append(node.func.id)
    return all_func


def extract_test_input(test_code, func_entry):
    assert_statements = extract_assert_statements(test_code)

    test_inputs = []
    for assert_statement in assert_statements:
        tree = ast.parse(assert_statement)
        all_args_nodes = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                all_args_nodes = parse_args(node, func_entry)
                break
        try:
            args = []
            for arg in all_args_nodes:
                args.append(eval(astor.to_source(arg)))
            test_inputs.append(args)
        except:
            continue
    return test_inputs


def extract_func_name(test_code):
    assert_statements = extract_assert_statements(test_code)

    all_funcs = []
    for assert_statement in assert_statements:
        tree = ast.parse(assert_statement)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                func_name = parse_func(node)
                all_funcs.append(func_name)
    return all_funcs


if __name__ == '__main__':
    tmp = \
        """
        def check(candidate):
            assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
            assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))
            assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))
        """

    extract_test_input(tmp)