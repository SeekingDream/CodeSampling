
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Tuple
from copy import deepcopy
from .io_types import CodeTask, CodeLLMOutput
from .abst_llm import AbstLLM, vLLM


class LLaMaLLM(vLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)
        self.prefix_sym = "<PRE>"
        self.suffix_sym = "<SUF>"
        self.mid_sym = "<MID>"
        self.stop = [
            "\n>>>", "\n$", '\nclass',
            '\ndef', '\n#', '\nprint',
            '\n}\n', "\n@",
            "\nif __name__ == '__main__':"
        ]

    def _task2prompt(self, task: CodeTask) -> str:
        # return self._lm_task2prompt(task)

        # return inst.format(lang=task.lang, code_prompt=task.prefix)
        return task.prefix


    def _prediction2output(self, prompt, task: CodeTask, model_prediction, cost_time) -> CodeLLMOutput:
        pred_code = model_prediction.outputs[0].text
        if task.task_name == "CC":
            raise NotImplementedError
        elif task.task_name == 'CG':
            code_blocks = task.prefix + pred_code
        else:
            raise NotImplementedError

        logits = [self.extract_token_prob(d) for d in model_prediction.outputs[0].logprobs]
        output = CodeLLMOutput(
            prompt_input=prompt,
            original_task=task,
            original_output=model_prediction,
            text=pred_code,
            logits=logits,
            final_code=code_blocks,
            cost_time=cost_time
        )
        return output, logits


