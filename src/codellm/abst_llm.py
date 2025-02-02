import ast
from typing import List
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
import time
from copy import deepcopy
from vllm.lora.request import LoRARequest
from vllm.distributed import destroy_distributed_environment
from vllm.distributed import destroy_model_parallel
import gc
import contextlib
import torch


from .io_types import CodeLLMOutput, CodeTask


class AbstLLM:
    NL_CG_PT = \
'''You are a helpful coding assistant producing high-quality code. 
Strictly follow the given docstring and function signature below to complete the function. 
Your code should always gracefully return. 
Your response should include all dependencies, headers and function declaration to be directly usable 
(even for the ones seen in the given part). 
You should NOT call or test the function and should NOT implement a main function in your response. 
{lang_instr}You should output your complete implementation in a single code block wrapped by triple backticks.

```{lang}
{code_prompt}
```
'''

    LANG_INSTR = {
        'python': 'You should implement the function in Python. ',
        'js': 'You should implement the function in JavaScript. ',
        'c': 'You should implement the function in pure C (NOT C++). ',
        'cpp': 'You should implement the function in C++ with C++ features as much as possible. ',
        'go': 'You should implement the function in Golang. ',
    }

    def __init__(self, model_name, is_lora):
        self.model_name = model_name
        self.is_lora = is_lora
        self.tokenizer = None

        self.temperature = None
        self.top_p = None
        self.max_tokens = None

        self.prefix_sym = None
        self.suffix_sym = None
        self.mid_sym = None
        self.mask_sym = None

        self.is_init = None
        self.stop = []

    def prompt2code(self, prompt: str) -> str:
        raise NotImplementedError

    def code_gen(self, task: CodeTask) -> CodeLLMOutput:
        return self.code_gen_batch([task])[0]

    def code_gen_batch(self, tasks: List[CodeTask]) -> List[CodeLLMOutput]:
        res = []
        for task in tasks:
            res.append(self.code_gen(task))
        return res

    def init_ai_kwargs(self, config):
        self.temperature = config['temperature']
        self.top_p = config['top_p']
        self.max_tokens = config['max_tokens']

    def _lm_task2prompt(self, task: CodeTask) -> str:
        lang = task.lang
        if task.task_name == "CC":
            raise NotImplementedError

        elif task.task_name == "CG":
            inst = deepcopy(self.NL_CG_PT)
            lang_instr = self.LANG_INSTR[lang]
            inst = inst.format(lang=lang, code_prompt=task.prefix, lang_instr=lang_instr)

            return inst

        else:
            raise NotImplementedError

    def extract_code_block(self, text):
        pattern = r"\{(\w+)\}\s*(.*?)\s*(?=\{|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)
        code_list = []
        for lang, code in matches:
            code_list.append(code)
        return "\n".join(code_list)



class vLLM(AbstLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)

        self.logprobs = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stop_token_ids = [self.tokenizer.eos_token_id]

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            enable_lora=self.is_lora,
            max_model_len=8192,
        )

        self.sampling_params = None
        self.dtype = None
        self.lora_request = None


    def init_ai_kwargs(self, config):
        super().init_ai_kwargs(config)
        # self.dtype = config["dtype"]
        lora_path = config['lora_path']
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop_token_ids=self.stop_token_ids,
            logprobs=self.logprobs,
            stop=self.stop

        )

        if lora_path is not None:
            self.lora_request = LoRARequest("tmp", 1, lora_path)
        self.is_init = True

    @staticmethod
    def extract_token_prob(d):
        tmp = list(d.values())[0]
        return tmp.decoded_token, tmp.logprob

    def _task2prompt(self, task: CodeTask) -> str:
        raise NotImplementedError

    def _prompt2output_batch(self, prompts: List[str]):
        
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=self.sampling_params,
            lora_request=self.lora_request
        )
        return outputs

    def _prediction2output(self, prompt, task: CodeTask, model_prediction, cost_time) -> CodeLLMOutput:
        pred_code = model_prediction.outputs[0].text
        if task.task_name == "CC":
            code_blocks = task.prefix + pred_code + task.suffix
        elif task.task_name == 'CG':
            code_blocks = task.prefix + pred_code
        else:
            raise NotImplementedError
        all_func_code = self.extract_all_func(code_blocks, task.lang)
        logits = [self.extract_token_prob(d) for d in model_prediction.outputs[0].logprobs]
        output = CodeLLMOutput(
            prompt_input=prompt,
            original_task=task,
            original_output=model_prediction,
            text=pred_code,
            logits=logits,
            final_code=all_func_code,
            cost_time=cost_time
        )
        return output

    def code_gen_batch(self, tasks: List[CodeTask]) -> List[CodeLLMOutput]:
        if not self.is_init:
            raise NotImplementedError
        prompts = [self._task2prompt(task) for task in tasks]
        t1 = time.time()
        outputs = self._prompt2output_batch(prompts)
        t2 = time.time()
        cost_time = t2 - t1
        res = [self._prediction2output(p, t, o, cost_time) for p, t, o in zip(prompts, tasks, outputs)]
        return res

    @staticmethod
    def cleanup():
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
