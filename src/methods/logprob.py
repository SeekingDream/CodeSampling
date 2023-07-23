from .abstract_selector import AbstractSelector
from .abstract_selector import AbstractExeSelector


class LogProbSelector(AbstractSelector):
    def __init__(self, all_res, approach_name):
        super(LogProbSelector, self).__init__(all_res)
        self.approach_name = approach_name
        self.dict_key = self.approach_name.split("-")[-1]

    def score_func(self, x):

        return x["not_degenerate"], x[self.dict_key]

        # sample_selection_function = lambda x: (
        #     x["not_degenerate"],
        #     x[criterion],
        # )


class LogProbEnsembleSelector(AbstractSelector):
    def __init__(self, all_res, approach_name: str):
        super(LogProbEnsembleSelector, self).__init__(all_res)
        self.approach_name = approach_name
        self.alpha = float(approach_name.split("#")[1])
        self.is_avg, self.is_sum = False, False

        if self.approach_name.startswith('avg'):
            self.is_avg = True
        elif self.approach_name.startswith('sum'):
            self.is_sum = True
        else:
            raise NotImplementedError

    def score_func(self, x):
        if self.is_avg:
            score = x["avg_reverse_logprob"] * self.alpha + x["avg_logprob"] * (1 - self.alpha)
            return x["not_degenerate"], score
        if self.is_sum:
            score = x["sum_reverse_logprob"] * self.alpha + x["sum_logprob"] * (1 - self.alpha)
            return x["not_degenerate"], score
        raise NotImplementedError


class ExeLogProbSelector(AbstractExeSelector):
    def __init__(self, all_res, approach_name, use_multi_assertions, good_execution_result):
        super(ExeLogProbSelector, self).__init__(all_res, use_multi_assertions, good_execution_result)
        self.approach_name = approach_name
        self.dict_key = self.approach_name.split("-")[-1]

    def score_func(self, x):
        res = (
            x["not_degenerate"],
            self.exe_func(x, self.good_execution_result),
            x[self.dict_key]
        )
        return res


class ExeLogProbEnsembleSelector(AbstractExeSelector):
    def __init__(self, all_res, approach_name, use_multi_assertions, good_execution_result):
        super(ExeLogProbEnsembleSelector, self).__init__(all_res, use_multi_assertions, good_execution_result)

        self.approach_name = approach_name
        self.alpha = float(approach_name.split("#")[1])

        self.is_avg, self.is_sum = False, False
        if self.approach_name.split('-')[1].startswith('avg'):
            self.is_avg = True
        elif self.approach_name.split('-')[1].startswith('sum'):
            self.is_sum = True
        else:
            raise NotImplementedError

    def score_func(self, x):
        exe_res = self.exe_func(x, self.good_execution_result)
        if self.is_avg:
            score = x["avg_reverse_logprob"] * self.alpha + x["avg_logprob"] * (1 - self.alpha)
            exe_res = self.exe_func(x, self.good_execution_result)
            return x["not_degenerate"], exe_res, score
        if self.is_sum:
            score = x["sum_reverse_logprob"] * self.alpha + x["sum_logprob"] * (1 - self.alpha)
            return x["not_degenerate"], exe_res, score
        raise NotImplementedError


