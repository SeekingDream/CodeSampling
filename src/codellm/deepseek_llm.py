
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from typing import List, Tuple
from tqdm import tqdm
import time

from .io_types import CodeTask, CodeLLMOutput
from .abst_llm import AbstLLM, vLLM


class CodeDeepSeekLLM(vLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)

        self.prefix_sym = "<｜fim▁begin｜>"
        self.suffix_sym = "<｜fim▁end｜>"
        self.mid_sym = "<｜fim▁hole｜>"
        self.stop = [
            "\n>>>", "\n$", '\nclass',
            '\ndef', '\n#', '\nprint',
            '\n}\n', "\n@",
            "\nif __name__ == '__main__':"
        ]

    def _task2prompt(self, task: CodeTask) -> str:
        if task.task_name == "CC":
            prompt = self.prefix_sym + task.prefix + self.mid_sym + task.suffix + self.suffix_sym
        elif task.task_name == "CG":
            prompt = task.prefix
        else:
            raise NotImplementedError
        return prompt


    def _prediction2output(self, prompt, task: CodeTask, model_prediction, cost_time) -> CodeLLMOutput:
        pred_code = model_prediction.outputs[0].text
        if task.task_name == "CC":
            code_blocks = task.prefix + pred_code + task.suffix
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
    #
    # def prompts2output(self, prompt: str) -> str:
    #     return self.prompt2output_batch([prompt])[0]
    #
    # def prompt2code(self, prompt: str) -> str:
    #     output = self.llm.generate(
    #         prompts=prompt,
    #         sampling_params=self.sampling_params
    #     )
    #     return output[0].outputs[0].text