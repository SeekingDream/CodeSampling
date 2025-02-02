import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq

from src.data import HumanEvalData, MBPPData
from src import CodeTask
from src.codellm import AbstLLM, vLLM, CodeLLaMaLLM, LLaMaLLM, CodeDeepSeekLLM
from src.description_mute import CharacterMutation, TokenMutation


SPLIT_SYM = "__SPLIT__"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OVERFIT_DIR = os.path.join(RESULTS_DIR, "overfit_dir")
os.makedirs(OVERFIT_DIR, exist_ok=True)
GENERATED_CODE_DIR = os.path.join(RESULTS_DIR, "generated_code")
os.makedirs(GENERATED_CODE_DIR, exist_ok=True)
FINAL_RES = "final_res"
os.makedirs(FINAL_RES, exist_ok=True)
NEW_PROMPT_DIR = os.path.join(RESULTS_DIR, "new_prompt")
os.makedirs(NEW_PROMPT_DIR, exist_ok=True)
EXE_RES_DIR = os.path.join(RESULTS_DIR, "exe_res_dir")
os.makedirs(EXE_RES_DIR, exist_ok=True)
PASS_AT_K_DIR = os.path.join(RESULTS_DIR, "pass_at_k")
os.makedirs(PASS_AT_K_DIR, exist_ok=True)


def load_dataset(data_id):
    if data_id == 0:
        dataset = HumanEvalData()
    elif data_id == 1:
        # load mbpp-sanitized
        dataset = MBPPData()
    elif data_id == 2:
        # load mbpp-full
        dataset = MBPPData(full=True)
    else:
        raise NotImplementedError
    return dataset


def model_id2name_cls(model_id: int):
    if model_id == 0:
        model_name = "meta-llama/Llama-3.2-1B"
        model_cls = LLaMaLLM
        is_lora = True
    elif model_id == 1:
        model_name = "meta-llama/Llama-3.2-3B"
        model_cls = LLaMaLLM
        is_lora = True
    elif model_id == 2:
        model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
        model_cls = CodeDeepSeekLLM
        is_lora = True
    elif model_id == 3:
        model_name = "meta-llama/Llama-3.1-8B"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 4:
        model_name = "meta-llama/CodeLlama-7b-hf"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 5:
        model_name = "meta-llama/CodeLlama-13b-hf"
        model_cls = LLaMaLLM
        is_lora = False
    elif model_id == 6:
        model_name = "deepseek-ai/DeepSeek-V2-Lite"
        model_cls = CodeDeepSeekLLM
        is_lora = False
    elif model_id == 7:
        model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
        model_cls = CodeDeepSeekLLM
        is_lora = False
    elif model_id == 8:
        model_name = "meta-llama/Llama-3.1-8B-Instruct"
        model_cls = LLaMaLLM
        is_lora = False
    else:
        raise ValueError(f"Model ID {model_id} is not valid")
    return model_name, model_cls, is_lora

def load_model(model_id):
    model_name, model_cls, is_lora = model_id2name_cls(model_id)
    model = model_cls(model_name, is_lora)

    model.model_name = model_name.split('/')[-1]
    return model


def load_perturbed_prompt(construct_prompt, task):
    if construct_prompt == "char_mutation":
        methods = CharacterMutation(task['prompt'], task['tests'], task['entry_point'])
        prompt = methods.mutate(task['language'])
    elif construct_prompt == "token_mutation":
        methods = TokenMutation(task['prompt'], task['tests'], task['entry_point'])
        prompt = methods.mutate(task['language'])
    else:
        raise NotImplementedError