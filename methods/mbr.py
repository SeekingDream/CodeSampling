import collections
import numpy as np

from .abstract_selector import AbstractSelector, AbstractExeSelector


class MBRSelector(AbstractExeSelector):
    # first assertion call matching
    @staticmethod
    def single_compare_function(x, y, good_execution_result=0):
        exec_x, exec_y = x["execution_result"], y["execution_result"]
        return MBRSelector.single_compare_matching(exec_x, exec_y, good_execution_result)

    @staticmethod
    def multi_compare_function(x, y, good_execution_result=0):
        exec_x, exec_y = x["execution_result_full"], y["execution_result_full"]
        assert len(exec_x) == len(exec_y)
        tmp = sum(
            [
                MBRSelector.single_compare_matching(single_x, single_y, good_execution_result)
                for single_x, single_y in zip(exec_x, exec_y)
            ]
        )
        return tmp == len(exec_x)

    @staticmethod
    def single_compare_matching(exec_x, exec_y, good_execution_result):
        try:
            if (
                    exec_x[0] == good_execution_result
                    and exec_y[0] == good_execution_result
                    and exec_x[1] == exec_y[1]
            ):
                return 1
            else:
                return 0
        except:
            return 0

    def __init__(self, all_res, use_multi_assertions, good_execution_result, approach_name):
        super(MBRSelector, self).__init__(all_res, use_multi_assertions, good_execution_result)

        self.quantile_threshold = None
        if not use_multi_assertions:
            self.compare_func = self.single_compare_function
        else:
            self.compare_func = self.multi_compare_function

        self.key_extractor = lambda x, y: all(
            [
                self.compare_func(x, y, self.good_execution_result),
                x["not_degenerate"],
                y["not_degenerate"],
            ]
        )
        self.second_key_extractor = lambda x: x["sum_logprob"]

    def init_filter(self, prob_id, candidate_ids):
        all_second_key = [self.second_key_extractor(self.data[idx][prob_id]) for idx in candidate_ids]
        threshold = np.quantile(all_second_key, self.quantile_threshold)

        filtered_ids = [idx for idx_i, idx in enumerate(candidate_ids) if all_second_key[idx_i] >= threshold]
        return filtered_ids

    def select_one_problem(self, prob_id, candidate_ids, sample_keys):
        max_key = None
        selected_item = None
        if self.quantile_threshold is not None:
            filtered_ids = self.init_filter(prob_id, candidate_ids)
        else:
            filtered_ids = candidate_ids

        for idx in filtered_ids:
            item = self.data[idx][prob_id]
            first_keys = list()
            # second_keys = list()
            for grndtruth_idx in filtered_ids:
                grndtruth_item = self.data[grndtruth_idx][prob_id]

                key = self.key_extractor(item, grndtruth_item)  # 每个item都有机会当grndtruth_item !!!差不多！！只要把key_extractor换了就行！
                first_keys.append(key)

            first_key = sum(first_keys)
            second_key = self.second_key_extractor(item)

            current_key = (first_key, second_key)

            sample_keys[idx].append(current_key)
            if max_key is None or current_key > max_key:
                max_key = current_key
                selected_item = item

        return selected_item, sample_keys

    def select(
            self, ids=None,
            return_keys=False
    ):
        if ids is None:
            ids = self.all_res.data.keys()
        ids = list(sorted(ids))

        sample_keys = collections.defaultdict(list)

        n_examples = len(self.data[ids[0]])
        selected_examples = list()
        for problem_id in range(n_examples):
            selected_item, sample_keys = self.select_one_problem(problem_id, ids, sample_keys)
            assert selected_item is not None
            selected_examples.append(selected_item)
        if return_keys:
            return selected_examples, sample_keys
        else:
            return selected_examples
