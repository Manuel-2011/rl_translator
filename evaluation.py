from torch.utils.data import Dataset, DataLoader
from logging import getLogger
import logging
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logfile_name = 'evaluation_exp9_qwen_notools.log'
# vllm_lora_adapter = 'models/tools_sft_rl_qwen7b_best_model'
vllm_lora_adapter = "models/best_policy_model13"
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
prompt_with_tools = False

start_time = time.time()

logger = getLogger(__name__)
logsdir = 'logs'
logpath = os.path.join(logsdir, logfile_name)
if os.path.exists(logpath):
  os.remove(logpath)
logging.basicConfig(filename=logpath, encoding='utf-8', level=logging.INFO)

class TextDataset(Dataset):
    def __init__(self, spa_path, wayuu_path):
        with open(spa_path, 'r', encoding='utf-8') as f:
            self.spa_lines = [line.strip() for line in f if line.strip()]

        with open(wayuu_path, 'r', encoding='utf-8') as f:
            self.wayuu_lines = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.spa_lines)

    def __getitem__(self, idx):
        spa = self.spa_lines[idx]
        wayuu = self.wayuu_lines[idx]
        
        return spa, wayuu
    
spanish_val_file = 'datasets/dev.es.txt'
wayuu_val_file = 'datasets/dev.guc.txt'

# Load the dataset
dataset = TextDataset(spanish_val_file, wayuu_val_file)

class CsvDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            titles = f.readline()
            self.lines = [line.strip().split(',')[:2] for line in f if len(line.strip().split(',')[:2]) == 2]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        spa, wayuu = self.lines[idx]
        
        return spa, wayuu
    
test_file = 'datasets/wayuu_dataset_test.csv'

# Load the dataset
test_dataset = CsvDataset(test_file)

import torch
import sacrebleu
from tqdm import tqdm
from vllm import SamplingParams

def get_rewards_translation(generations, correct_translations):

    bleu = sacrebleu.BLEU(effective_order = True)
    def get_bleu_score(sample, correct_translation):
        # Compute bleu score for each sample. 
        # Bleu score normalized to [0, 1]
        return bleu.sentence_score(sample, 
                                   [correct_translation]
                                   ).score

    answer_bleu_scores = [
        get_bleu_score(sample, translation)
        for sample, translation in zip(generations, correct_translations)
    ]
    
    return answer_bleu_scores

translate_prompt_template_tool="""Translate the following Spanish text into Wayuunaiki.
Begin by identifying any words or phrases you're unsure how to translate. Then, you may look up those words using the dictionary tool by wrapping the Spanish word in <spa_to_wayuu> and </spa_to_wayuu>,
and doind that for every unknown word. The dictionary will return matches enclosed in <matches> and </matches>. You can use the dictionary as many times as necessary.
Once you have all the information you need, provide the final translation enclosed in <answer> and </answer>. For example: <answer> xxx </answer>.

Spanish text: {}"""

translate_prompt_template="""Translate the following Spanish text into Wayuunaiki. Provide the final translation enclosed in <answer> and </answer>. For example: <answer> xxx </answer>.
Spanish text: {}"""

if prompt_with_tools:
    custom_prompt_template = translate_prompt_template_tool
else:
    custom_prompt_template = translate_prompt_template

def generate_batch_completion(model, tokenizer, prompts: list, actions_num=1, custom_prompt_template=None, **kwargs):
    prompt_template = custom_prompt_template if custom_prompt_template else translate_prompt_template_tool
    batch = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_template.format(prompt)}
    ] for prompt in prompts]
    texts = tokenizer.apply_chat_template(
        batch,
        tokenize=False,
        add_generation_prompt=True,
    )

    default_sampling_args = {
        'max_new_tokens': 512,
        'temperature': 0.8,
        'top_p': 0.95,
    }
    default_sampling_args.update(kwargs)

    model_inputs = tokenizer(texts)

    inputs = model_inputs.input_ids
    dones = [False] * len(prompts)
    prompt_length = [len(input_ids) for input_ids in inputs]
    mask = [[1] * len(input_ids) for input_ids in inputs]
    responses = [""] * len(prompts)
    tools_enabled = kwargs.get('tools', [])
    tools_enabled = [] if tools_enabled is None else tools_enabled
    stop_tokens = [tool['end_token'] for tool in tools_enabled]
    tool_used = [False] * len(prompts)
    how_many_tool_calls = [0] * len(prompts)
    unfinished_answers = 0
    for action_step in range(actions_num + 1 if len(tools_enabled) > 0 else 1):
        sampling_params = SamplingParams(temperature=default_sampling_args["temperature"], top_p=default_sampling_args['top_p'], top_k=-1, max_tokens=default_sampling_args['max_new_tokens'],
            stop=stop_tokens)
        outputs = model.generate(prompt_token_ids=inputs, sampling_params=sampling_params, lora_request=kwargs['lora_request'], use_tqdm=False)

        for j, output in enumerate(outputs):
            if dones[j]:
                continue
            
            for tool in tools_enabled:
                if output.outputs[0].stop_reason == tool['end_token'] and tool['start_token'] in output.outputs[0].text:
                    api_args = output.outputs[0].text.split(tool['start_token'])[1].strip()
                    api_result = tool['api'](api_args)
                    # responses[j] += f"{tool['start_token']} " + api_args + f" {tool['end_token']}" + api_result
                    responses[j] += output.outputs[0].text + f"{tool['end_token']}" + api_result
                    api_result_tokens = tokenizer.encode(api_result, return_tensors=None)
                    inputs[j] += list(output.outputs[0].token_ids) + api_result_tokens

                    tool_used[j] = True
                    how_many_tool_calls[j] += 1
                    break # Only one tool can be used at a time
            if output.outputs[0].finish_reason == "stop" and output.outputs[0].stop_reason is None:
                responses[j] += output.outputs[0].text
                inputs[j] += list(output.outputs[0].token_ids)
                dones[j] = True
            elif output.outputs[0].stop_reason not in stop_tokens:
                # print(f"Unexpected finish reason: {output.outputs[0].finish_reason} {output.outputs[0].stop_reason}")
                unfinished_answers += 1
                responses[j] += tokenizer.eos_token
                inputs[j] += [tokenizer.eos_token_id]
                dones[j] = True
        
        if all(dones):
            break

    return responses, tool_used, how_many_tool_calls, unfinished_answers

import re

def extract_answer(response, transform_fn = lambda x: x, nan_val = None)->str|None:
    ans = re.match('.*?<answer>(.*?)</answer>\.?\s*$', response, re.DOTALL|re.MULTILINE)
    if ans:
        try:
            return transform_fn(ans[1].strip())
        except:
            return nan_val
    return nan_val

def evaluate_model(model, tokenizer, dataloader, actions_num=1, lora_request=None, tools=None, custom_prompt_template=None):
    sum_bleu = 0
    num_samples = 0
    tools_used_in_total = 0
    calls_per_sample = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs, targets = batch

            # Generate translations
            outputs, tools_used, how_many_tool_calls, unfinished_answers = generate_batch_completion(model, tokenizer, inputs, actions_num=actions_num, lora_request=lora_request, tools=tools, temperature=0, top_p=1, max_new_tokens=768, custom_prompt_template=custom_prompt_template)

            tools_used_in_total += sum(tools_used)
            calls_per_sample += sum(how_many_tool_calls)

            generated_translations = [
                extract_answer(output, transform_fn=lambda x: x.strip(), nan_val='')
                for output in outputs
            ]
            # Calculate BLEU scores
            bleu_scores = get_rewards_translation(generated_translations, targets)
            
            sum_bleu += sum(bleu_scores)
            num_samples += len(bleu_scores)
    avg_bleu = sum_bleu / num_samples if num_samples > 0 else 0
    tools_used_avg = tools_used_in_total / num_samples
    calls_per_sample_avg = calls_per_sample / tools_used_in_total if tools_used_in_total > 0 else 0
    unfinished_answers_avg = unfinished_answers / num_samples if num_samples > 0 else 0
    return avg_bleu, tools_used_avg, calls_per_sample_avg, unfinished_answers_avg

def spa_to_wayu_dictionary(spanish_word, max_matches=5):
    dictionary_path = 'assets/spanish_to_wayuunaiki_short.csv'

    with open(dictionary_path, 'r', encoding='utf-8') as f:
        all_matches = []
        line = f.readline()
        while line != '' and len(all_matches) < max_matches:
            data = line.strip().split(',')
            if re.search(rf'\b{re.escape(spanish_word)}\b', data[0], re.IGNORECASE):
                all_matches.append(data)
            line = f.readline()

    if len(all_matches) > 0:
        result = " <matches> " + '\n'.join(f'{spa}: {wayuu}' for spa, wayuu in all_matches) + " </matches>"
        # print(f'CORRECT USE OF SPA_TO_WAYU TOOL. Word: {spanish_word}, Result: {result}')
    else:
        result = " <matches> No matches found </matches>"
        # print(f'NO_MATCHES SPA_TO_WAYU TOOL. Word: {spanish_word}')

    return result

TOOLS = [
    {
        'name': 'spa_to_wayu',
        'description': 'A tool that translates a word from Spanish to Wayuunaiki.',
        'api': spa_to_wayu_dictionary,
        'start_token': '<spa_to_wayuu>',
        'end_token': '</spa_to_wayuu>',
    }
]

# from pretrained peft model
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


tokenizer = AutoTokenizer.from_pretrained(base_model_name)

inference_engine = LLM(
    model=base_model_name,
    enable_lora=True,
    max_lora_rank=64,
    max_loras=1,
    gpu_memory_utilization=0.2, # CHANGE
    # enable_prefix_caching=True,
    swap_space=6,
    scheduling_policy="fcfs",
    dtype=torch.bfloat16,
    max_model_len=2060,
    # enable_sleep_mode=True,
    )

if vllm_lora_adapter:
    lora_request=LoRARequest('adapter', 1, vllm_lora_adapter)
else:
    lora_request=None

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Evaluate the model
logger.info(f'Validation dataset')
avg_bleu, tools_used_avg, calls_per_sample_avg, unfinished_avg = evaluate_model(inference_engine, tokenizer, dataloader, actions_num=4, lora_request=lora_request, tools=TOOLS, custom_prompt_template=custom_prompt_template)
logger.info(f"Average BLEU score: {avg_bleu:.4f}")
logger.info(f"Average tools used: {tools_used_avg:.4f}")
logger.info(f"Average calls per sample: {calls_per_sample_avg:.4f}")
logger.info(f"Average unfinished answers: {unfinished_avg:.4f}")

dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Evaluate the model
logger.info(f'Test dataset')
avg_bleu, tools_used_avg, calls_per_sample_avg, unfinished_avg = evaluate_model(inference_engine, tokenizer, dataloader, actions_num=4, lora_request=lora_request, tools=TOOLS, custom_prompt_template=custom_prompt_template)
logger.info(f"Average BLEU score: {avg_bleu:.4f}")
logger.info(f"Average tools used: {tools_used_avg:.4f}")
logger.info(f"Average calls per sample: {calls_per_sample_avg:.4f}")
logger.info(f"Average unfinished answers: {unfinished_avg:.4f}")

end_time = time.time()
execution_time = end_time - start_time
logger.info(f'Total execution time in minutes: {execution_time/60:.2f}')