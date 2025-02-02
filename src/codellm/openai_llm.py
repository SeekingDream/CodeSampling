import openai
from openai import OpenAI
from typing import List, Tuple
from time import sleep
from tqdm import tqdm
from .io_types import CodeTask, CodeLLMOutput
from .abst_llm import AbstLLM



class OpenAILLM(AbstLLM):
    def __init__(self, model_name, is_lora):
        super().__init__(model_name, is_lora)
        key = config['openai']
        self.client = OpenAI(
            api_key=key,
        )
        self.mask_token = "<|mask|>"

        self.client_kwargs = {
            "model": model_name,
            "temperature": config['temperature'],
            "max_tokens": config['max_tokens'],
            "logprobs": True
            # "top_p": args.top_p,
            # "frequency_penalty": 0,
            # "presence_penalty": 0,
            # "n": args.n,
            # "timeout": args.openai_timeout,
            # "stop": args.stop, --> stop is only used for base models currently
        }

    def _task2prompt(self, task: CodeTask) -> str:
        return self._lm_task2prompt(task)

    def prediction2output(self, prompt, task: CodeTask, model_prediction) -> CodeLLMOutput:
        logits = [(d.token, d.logprob) for d in model_prediction.logprobs.content]
        pred_code = model_prediction.message.content
        final_code = extract_code_blocks(pred_code)
        final_code = self.extract_all_func(final_code, task.lang)
        output = CodeLLMOutput(
            prompt_input=prompt,
            original_task=task,
            model_config=self.config,
            original_output=model_prediction,
            text=pred_code,
            logits=logits,
            final_code=final_code,
            cost_time=None,
        )
        return output, logits

    def code_gen(self, task: CodeTask):
        prompt = self._task2prompt(task)

        output = self.prompt2output(prompt)
        return self.prediction2output(prompt, task, output)

    def prompt2output(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.chat.completions.create(
                messages=messages,
                **self.client_kwargs,
            )
        except (
                openai.APIError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.OpenAIError,
                openai.APIStatusError,
                openai.APITimeoutError,
                openai.InternalServerError,
                openai.APIConnectionError,
        ) as e:
            print("Exception: ", repr(e))
            print("Sleeping for 30 seconds...")
            print("Consider reducing the number of parallel processes.")
            sleep(30)
            return self.prompt2output(prompt)
        except Exception as e:
            print(f"Failed to run the model for")
            print("Exception: ", repr(e))
            raise e
        return response.choices[0]

    def code_gen_batch(self, tasks: List[CodeTask]) -> List[CodeLLMOutput]:
        res = []
        for task in tqdm(tasks):
            res.append(self.code_gen(task))
        return res
