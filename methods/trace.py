import random

import numpy as np
import regex
import collections
from typing import List
# from sklearn.metrics.pairwise import cosine_similarity


from .abstract_selector import AbstractExeSelector


def compute_similarity(vec_list, target_vec):
    similarities = []
    for vec in vec_list:
        similarity = cosine_similarity([target_vec], [vec])[0][0]
        similarities.append(similarity)
    return similarities


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
            '\x1b[1mIndexError', '\x1b[1mTypeError'
        ]:
            return_str = 'error:: ' + parsed_l[1]
            trace.append(return_str)
        elif parsed_l[1] in ['path:...', 'call', 'time:', 'value:..', 'exception']:
            continue
        else:
            print(parsed_l[1])
            continue
    return trace


def trace2str(trace: List[str]):
    new_trace = [d.replace('\x1b[0m', '') for d in trace]
    return ';\n'.join(new_trace)


class TraceSelector(AbstractExeSelector):
    def __init__(self, all_res, use_multi_assertions, good_execution_result):
        super(TraceSelector, self).__init__(all_res, use_multi_assertions, good_execution_result)
        self.model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
        self.model = self.model.to('cuda').eval()
        self.all_trace = all_res.traces  #self.get_parsed_trace()

    # def get_parsed_trace(self):
    #     traces = {}
    #     for idx in self.data:
    #         post_trace_list = []
    #         for problem_id, item in enumerate(self.data[idx]):
    #             post_traces = [process_trace(raw_trace) for raw_trace in item['raw_trace']]
    #             post_trace_list.append(post_traces)
    #         traces[tuple2str(idx)] = post_trace_list
    #     return traces

    def compute_trace_sim(self, trace_items):
        all_vecs = []
        for traces in trace_items:
            vecs = []
            for trace in traces:
                vec = self.model.encode(trace2str(trace))
                vecs.append(vec.reshape([1, -1]))
            all_vecs.append(vecs)
        new_all_vecs = list(zip(*all_vecs))
        center_list, distance_list = [], []
        for vecs in new_all_vecs:
            vecs = np.concatenate(vecs, axis=0)
            center = vecs.mean(0)
            distance = compute_similarity(vecs, center)

            distance_list.append(np.array(distance).reshape([-1, 1]))
            center_list.append(center_list)
        return center_list, distance_list

    def output_score_func(self, candidate_items):
        majority_vote = {}
        for candidate in candidate_items:
            if self.use_multi_assertions:
                key = str([d[1] for d in candidate["execution_result_full"]])
            else:
                key = str(candidate['execution_result'][1])
            if key not in majority_vote:
                majority_vote[key] = 0
            majority_vote[key] += 1
        majority_vote = sorted(majority_vote.items(), key=lambda kv:(kv[1], kv[0]))
        pred_oracle = majority_vote[-1][0]
        output_scores = []
        for candidate in candidate_items:
            if self.use_multi_assertions:
                key = str([d[1] for d in candidate["execution_result_full"]])
            else:
                key = str(candidate['execution_result'][1])
            if key == pred_oracle:
                output_scores.append(1)
            else:
                output_scores.append(0)
        return np.array(output_scores)

    def trace_score_func(self, traces_list):
        scores = []
        for traces in traces_list:
            current_s = []
            for trace in traces:
                command = '\n'.join(trace)
                if 'return' not in command:
                    current_s.append(False)
                else:
                    current_s.append(True)
            if False in current_s:
                scores.append(0)
            else:
                scores.append(1)
        return np.array(scores)

    def select_one_problem(self, candidate_items, traces_list):
        output_scores = self.output_score_func(candidate_items)
        exe_scores = np.array([self.exe_func(x, self.good_execution_result) for x in candidate_items])
        log_scores = np.array([x['avg_logprob'] for x in candidate_items])
        trace_scores = self.trace_score_func(traces_list)

        total_scores = trace_scores * 100 + exe_scores * 10 + output_scores + log_scores
        index = np.argmax(total_scores)
        return candidate_items[index]

        # center_list, distance_list = self.compute_trace_sim(trace_items)
        # if len(distance_list) != 0:
        #     if self.use_multi_assertions:
        #         distance = np.sum(np.concatenate(distance_list, axis=1), axis=1)
        #     else:
        #         distance = distance_list[0]
        #     index = np.argmax(distance)
        #     return candidate_items[index]
        # else:
        #     return random.choice(candidate_items)

    def construct_init_candidate(self, problem_id, ids):
        candidate_items, trace_items = [], []
        for idx in ids:
            current_trace = self.all_trace[tuple2str(idx)][problem_id]
            if None in current_trace:
                continue
            candidate_items.append(self.all_res.data[idx][problem_id])
            trace_items.append(current_trace)
        return candidate_items, trace_items

    def select(
            self, ids=None,
            return_keys=False
    ):
        if ids is None:
            ids = self.all_res.data.keys()
        ids = list(sorted(ids))
        n_examples = len(self.all_res.data[ids[0]])
        selected_examples = list()
        sample_scores = collections.defaultdict(list)
        for problem_id in range(n_examples):

            candidate_items, trace_items = self.construct_init_candidate(problem_id, ids)

            if len(candidate_items):
                selected_item = self.select_one_problem(candidate_items, trace_items)
            else:
                random_idx = random.choice(ids)
                selected_item = self.all_res.data[random_idx][problem_id]

            selected_examples.append(selected_item)
        if return_keys:
            return selected_examples, sample_scores
        else:
            return selected_examples