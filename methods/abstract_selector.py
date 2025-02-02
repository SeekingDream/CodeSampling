import os
import pickle
import numpy as np
from glob import glob
from tqdm import tqdm
import collections
import json
import regex

from pyminifier import analyze
from pyminifier.minification import remove_comments_and_docstrings, remove_blank_lines


def clean_comment(code):
    code = remove_comments_and_docstrings(code)
    code = remove_blank_lines(code)
    return code


def filter_empty(x, remove_function_header=False):
    code = x["trg_prediction"]
    if remove_function_header:
        code = "\n".join(
            [l for l in code.split("\n") if not l.strip().startswith("def")]
        )
    try:
        code = clean_comment(code)
    except:
        code = ""
    return code.strip() not in ["", "pass", "return"]


def filter_repeat(x, threshold=0.25):
    import zlib

    bytes_x = bytes(x, encoding="utf-8")
    comp_x = zlib.compress(bytes_x)
    return len(comp_x) / len(bytes_x) > threshold


def tuple2str(a):
    a_list = [str(kkk) for kkk in a]
    return '____'.join(a_list)


def process_trace(raw_trace):
    var_pattern = regex.compile("[a-zA-Z_][a-zA-Z0-9_]* = .*")

    trace = []
    lines = raw_trace[-1]
    if lines is None:
        return None
    parsed_lines = [l.strip().split(' ') for l in lines]
    for parsed_l, line in zip(parsed_lines, lines):

        if parsed_l[1].startswith('var:'):
            if regex.search(var_pattern, line) is not None:
                trace.append("var:: " + regex.search(var_pattern, line).group())
        elif parsed_l[1] == 'line':
            statement = line.split('\x1b[0m')[-1]
            trace.append("line:: " + statement)
        elif parsed_l[1] == 'return':
            if 'pass' in line.split('return')[-1].strip():
                continue
            return_str = 'return:: ' + line.split('return')[-1].strip()
            trace.append(return_str)
        elif parsed_l[1] in [
            '\x1b[1mValueError:', '\x1b[1mNameError:',
            '\x1b[1mIndexError:', '\x1b[1mTypeError:',
            '\x1b[1mValueError', '\x1b[1mNameError',
            '\x1b[1mIndexError', '\x1b[1mTypeError',
            "\x1b[1mTypeError\x1b[0m", '\x1b[1mKeyError:',

        ]:
            return_str = 'error:: ' + parsed_l[1]
            trace.append(return_str)
        elif parsed_l[1] in ['path:...', 'call', 'time:', 'value:..', 'exception']:
            continue
        else:
            print(parsed_l[1])
            continue
    return trace


class ExecutionDatabase:
    def __init__(
            self,
            paths,
            split="dev",
            execution_type=None,
            tag="",
            model="",
            verbose=False,
            dataset="",
            no_rejection=False,
    ):
        self.paths = (
            list(sorted(glob(paths, recursive=True)))
            if isinstance(paths, str)
            else list(sorted(paths))
        )
        self.split = split
        self.data = collections.defaultdict(list)
        self.args = collections.defaultdict(list)
        self.tag = tag
        self.model = model
        self.dataset = dataset
        self.verbose = verbose
        for i, path in tqdm(
                enumerate(self.paths),
                total=len(self.paths),
                desc="loading jsons",
                disable=not self.verbose,
        ):
            self.args[i] = pickle.load(open(f"{self.paths[0]}/configs.pkl", "rb"))
            idx = 0
            if self.tag != "":
                file_path = f"{path}/{split}-{idx}-{tag}.jsonl"
            else:
                file_path = f"{path}/{split}-{idx}.jsonl"
            while os.path.exists(file_path):
                self.data[i, idx].extend([json.loads(x) for x in open(file_path)])
                idx += 1
                if self.tag != "":
                    file_path = f"{path}/{split}-{idx}-{tag}.jsonl"
                else:
                    file_path = f"{path}/{split}-{idx}.jsonl"

        print(f"{len(self.data)} cached samples")

        for path_id, sample_id in tqdm(
                self.data, desc="loading logprobs", disable=not self.verbose
        ):

            if (
                    self.paths[path_id].find("nl2bash") != -1
            ):  # NL2bash data, exec simulation

                if self.tag != "":
                    file_name = f"{path}/{split}-{sample_id}-{tag}"
                else:
                    file_name = f"{path}/{split}-{sample_id}"
                exec_results = pickle.load(open(f"{file_name}.exec.pkl", "rb"))
                simulate_exec_results = pickle.load(
                    open(f"{file_name}.exec.simulate.pkl", "rb")
                )
                splitted_exec_results = pickle.load(
                    open(f"{file_name}.exec.splitted.pkl", "rb")
                )
                char_bleus = pickle.load(open(f"{file_name}.exec.bleu.pkl", "rb"))

            for item_i, item in enumerate(self.data[path_id, sample_id]):
                if no_rejection:
                    item["not_degenerate"] = True
                else:
                    # implementing degenerate solution rejection
                    if self.dataset in ["codet_humaneval", "mbpp_sanitized", "mbpp_hyx"]:
                        item["not_degenerate"] = filter_empty(
                            item, remove_function_header=False
                        ) and filter_repeat(item["trg_prediction"])
                    elif self.dataset in ["mbpp"]:
                        item["not_degenerate"] = filter_empty(
                            item, remove_function_header=True
                        )
                    elif self.dataset in ["spider", "nl2bash"]:
                        item["not_degenerate"] = len(
                            item["trg_prediction"]
                        ) != 0 and filter_repeat(item["trg_prediction"])
                    else:
                        raise ValueError("Invalid Dataset.")
                avg_logprob, sum_logprob = self.extract_logprob_stats(item, path_id)
                item["avg_logprob"] = avg_logprob
                item["sum_logprob"] = sum_logprob

                reverse_logprob = self.extract_reverse_logprob(item, path_id)
                (
                    item["sum_reverse_logprob"],
                    item["avg_reverse_logprob"],
                ) = reverse_logprob

                if (
                        self.paths[path_id].find("nl2bash") != -1
                ):  # NL2bash data, exec simulation
                    item["executable"] = exec_results[item_i]
                    item["trg_prediction_splitted"] = splitted_exec_results[item_i]
                    item["execution_result_simulated"] = simulate_exec_results[item_i]
                    item["charbleu"] = char_bleus[item_i]

        self.execution_type = execution_type
        self.load_execution()
        # self.traces = self.get_parsed_trace()

    def get_parsed_trace(self):
        traces = {}
        for idx in self.data:
            post_trace_list = []
            for problem_id, item in enumerate(self.data[idx]):
                post_traces = [process_trace(raw_trace) for raw_trace in item['raw_trace']]
                post_trace_list.append(post_traces)
            traces[tuple2str(idx)] = post_trace_list
        return traces

    def extract_reverse_logprob(self, item, path_id):
        if "prompt_reverse_logprobs" not in item:
            return 0, 0
        logprobs = item["prompt_reverse_logprobs"]
        return np.sum(logprobs), np.mean(logprobs)

    def extract_logprob_stats(self, item, path_id):
        current_seq = ""
        if "codex" in self.model:
            extracted_position = None
            for i, _ in enumerate(item["tokens"]):
                current_seq += item["tokens"][i]
                end_template = self.args[path_id].end_template
                if isinstance(end_template, list):
                    end_template = ""
                if (
                        current_seq.find(item["trg_prediction"]) != -1
                        and current_seq.find(end_template) != -1
                ):
                    extracted_position = i + 1
                    break
            logprobs = (
                item["logprobs"][:extracted_position]
                if extracted_position is not None
                else item["logprobs"]
            )
            logprobs = list(
                filter(lambda x: x < 0, logprobs)
            )  # handle potential codex bug on positive log probability
        else:
            logprobs = item["logprobs"]
        return np.mean(logprobs), np.sum(logprobs)

    def load_execution(self):
        '''
        这个函数的主要功能是根据指定的文件名模式，从指定路径中加载执行结果，并将其存储在给定的数据字典中。
        根据文件名的不同，执行结果存储在数据字典的不同字段中。
        '''

        exec_list = [
            ("exec.pkl", "execution_result"),
            ("execfull.pkl", "execution_result_full"),
            ("execfullpass.pkl", "execution_result_full_pass"),
            # ("raw_trace.pkl", "raw_trace"),
            # ("gen.execfull.pkl", "gen_execution_result_full"),
        ]
        for suffix, result_name in exec_list:
            for i, idx in self.data:
                if self.tag == "":
                    out_name = f"{self.split}-{idx}.{suffix}"
                else:
                    out_name = f"{self.split}-{idx}-{self.tag}.{suffix}"
                path = self.paths[i]

                if os.path.exists(f"{path}/{out_name}"):
                    execution_results = pickle.load(open(f"{path}/{out_name}", "rb"))

                while len(execution_results) < len(self.data[i, idx]):
                    execution_results.append(execution_results[-1])

                assert len(execution_results) == len(self.data[i, idx])
                for j, execution_result in enumerate(execution_results):
                    self.data[i, idx][j][result_name] = execution_result


class AbstractSelector:
    def __init__(self, all_res: ExecutionDatabase):
        self.all_res = all_res
        self.data = self.all_res.data

    def rank_code(self):
        pass

    @staticmethod
    def score_func(x):
        pass

    def select(
            self, ids=None,
            return_keys=False
    ):
        if ids is None:
            ids = self.all_res.data.keys()
        ids = list(sorted(ids))
        n_examples = len(self.all_res.data[ids[0]])
        selected_examples = list()
        sample_keys = collections.defaultdict(list)
        for i in range(n_examples):
            max_key = None
            selected_item = None
            for idx in ids:
                item = self.all_res.data[idx][i]
                key = self.score_func(item)
                sample_keys[idx].append(key)
                if max_key is None or key > max_key:
                    max_key = key
                    selected_item = item
            assert selected_item is not None
            selected_examples.append(selected_item)
        if return_keys:
            return selected_examples, sample_keys
        else:
            return selected_examples


def exe_selection_function(x, good_execution_result=0):
    exec_res = x["execution_result"]
    return exec_res[0] == good_execution_result


def multi_exe_selection_function(x, good_execution_result=0):
    exec_res = x["execution_result_full"]
    return sum([e[0] == good_execution_result for e in exec_res])


class AbstractExeSelector(AbstractSelector):
    def __init__(self, all_res, use_multi_assertions, good_execution_result):
        super(AbstractExeSelector, self).__init__(all_res)
        self.use_multi_assertions = use_multi_assertions
        self.good_execution_result = good_execution_result

        if not use_multi_assertions:
            self.exe_func = exe_selection_function
        else:
            self.exe_func = multi_exe_selection_function


