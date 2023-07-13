
from .abstract_selector import AbstractSelector


class OracleSelector(AbstractSelector):
    def __init__(self, all_res):
        super(OracleSelector, self).__init__(all_res)

    def rank_code(self):
        pass

    @staticmethod
    def score_func(x):
        if isinstance(x["execution_result_full_pass"], bool):
            return int(x["execution_result_full_pass"])
        elif isinstance(x["execution_result_full_pass"], list):
            return int(
                all(
                    isinstance(exec_result[1], bool) and exec_result[1] == True
                    for exec_result in x["execution_result_full_pass"]
                )
            )
        else:
            raise NotImplementedError
