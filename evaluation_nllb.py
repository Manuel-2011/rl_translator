from torch.utils.data import Dataset, DataLoader
from logging import getLogger
import logging
import os
import time
from transformers.tokenization_utils import AddedToken
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

logfile_name = 'evaluation_exp14_nllb.log'
# vllm_lora_adapter = 'models/tools_sft_rl_qwen7b_best_model'
lora_adapter = "models/grpo_policy_model_nllb"
# lora_adapter = None
base_model_name = "models/nllb_wayuu_esp_completo_1_3B-V2"
prompt_with_tools = False
src_lang = "spa_Latn"
tgt_lang = "way_Latn"

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


def generate_batch_completion(model, tokenizer, prompts: list, return_ids=False, **kwargs):
    default_sampling_args = {
        'do_sample': False, # FIXME not enough memory in local
        'max_new_tokens': 768,
        'temperature': 0.8,
        'top_p': 0.95, # FIXME not enough memory in local
    }
    default_sampling_args.update(kwargs)

    model_inputs = tokenizer(prompts, padding='longest', padding_side='left', \
        return_tensors="pt").to(model.device) # No VLLM
    
    outputs = model.generate(
        inputs=model_inputs.input_ids,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("way_Latn"), # FIXME convert to param
        **default_sampling_args
    ) # Generation no VLLM

    if return_ids:
        generation_ids = [model_inputs.input_ids.tolist()[0] + list(output) for output in outputs.tolist()]  # Diferent tokenizer model.inputs
        # padding the generation_ids to the max length
        max_length = max([len(ids) for ids in generation_ids])
        generation_ids = [ids + [tokenizer.pad_token_id]*(max_length-len(ids)) for ids in generation_ids]
        generation_ids = torch.tensor(generation_ids)
        return generation_ids, len(model_inputs.input_ids[0])

    completions = tokenizer.batch_decode(outputs, skip_special_tokens=True) # No text in outputs had to tokenize decode
    return completions

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
            output = generate_batch_completion(model, tokenizer, inputs, temperature=0, top_p=1, max_new_tokens=512)

            # Calculate BLEU scores
            bleu_scores = get_rewards_translation(output, targets)
            
            sum_bleu += sum(bleu_scores)
            num_samples += len(bleu_scores)

            del output
            del bleu_scores
            del inputs
            del targets
            torch.cuda.empty_cache()
    avg_bleu = sum_bleu / num_samples if num_samples > 0 else 0
    return avg_bleu

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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="cuda:2"
    )
tokenizer = AutoTokenizer.from_pretrained(base_model_name, src_lang=src_lang)#, tgt_lang=tgt_lang)
tokenizer.add_tokens(AddedToken(tgt_lang, normalized=False, special=True))
if lora_adapter:
    inference_engine = PeftModel.from_pretrained(model, lora_adapter)
else:
    inference_engine = model

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Evaluate the model
logger.info(f'Validation dataset')
avg_bleu = evaluate_model(inference_engine, tokenizer, dataloader)
logger.info(f"Average BLEU score: {avg_bleu:.4f}")

dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Evaluate the model
logger.info(f'Test dataset')
avg_bleu = evaluate_model(inference_engine, tokenizer, dataloader)
logger.info(f"Average BLEU score: {avg_bleu:.4f}")

end_time = time.time()
execution_time = end_time - start_time
logger.info(f'Total execution time in minutes: {execution_time/60:.2f}')