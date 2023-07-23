import random

import numpy as np
import regex
import collections
from sentence_transformers import SentenceTransformer
from typing import List
from sklearn.metrics.pairwise import cosine_similarity


from .abstract_selector import AbstractSelector


def tuple2str(a):
    a_list = [str(kkk) for kkk in a]
    return '____'.join(a_list)


def compute_similarity(vec_list, target_vec):
    similarities = []
    for vec in vec_list:
        similarity = cosine_similarity([target_vec], [vec])[0][0]
        similarities.append(similarity)
    return similarities


def process_trace(raw_trace):
    pattern = regex.compile("[a-zA-Z_][a-zA-Z0-9_]* = .*")
    trace = []
    lines = raw_trace[-1]
    if lines is None:
        return None
    for line in lines:
        # print(line)
        if 'var' in line and regex.search(pattern, line) is not None:
            # print(regex.search(pattern, line).xgroup())
            trace.append(regex.search(pattern, line).group())
    return trace


def trace2str(trace: List[str]):
    new_trace = [d.replace('\x1b[0m', '') for d in trace]
    return ';\n'.join(new_trace)


class TraceSelector(AbstractSelector):
    def __init__(self, all_res, use_multi_assertions):
        super(TraceSelector, self).__init__(all_res)
        self.model = SentenceTransformer("flax-sentence-embeddings/st-codesearch-distilroberta-base")
        self.model = self.model.to('cuda').eval()
        self.use_multi_assertions = use_multi_assertions
        self.all_trace = self.get_parsed_trace()

    def get_parsed_trace(self):
        traces = {}
        for idx in self.data:
            post_trace_list = []
            for problem_id, item in enumerate(self.data[idx]):
                post_traces = [process_trace(raw_trace) for raw_trace in item['raw_trace']]
                post_trace_list.append(post_traces)
            traces[tuple2str(idx)] = post_trace_list
        return traces

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

    def select_one_problem(self, candidate_items, trace_items):
        center_list, distance_list = self.compute_trace_sim(trace_items)
        if len(distance_list) != 0:
            if self.use_multi_assertions:
                distance = np.sum(np.concatenate(distance_list, axis=1), axis=1)
            else:
                distance = distance_list[0]
            index = np.argmax(distance)
            return candidate_items[index]
        else:
            return random.choice(candidate_items)

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