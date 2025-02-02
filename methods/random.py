import random
from .abstract_selector import AbstractSelector
from .abstract_selector import AbstractExeSelector


class RandomSelector(AbstractSelector):
    def __init__(self, all_res):
        super(RandomSelector, self).__init__(all_res)

    def score_func(self, x):
        return x["not_degenerate"], random.random()


class ExeRandomSelector(AbstractExeSelector):
    def __init__(self, all_res, use_multi_assertions, good_execution_result):
        super(ExeRandomSelector, self).__init__(all_res, use_multi_assertions, good_execution_result)

    def score_func(self, x):
        return x["not_degenerate"], self.exe_func(x, self.good_execution_result), random.random()
