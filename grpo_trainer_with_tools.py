# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from logging import getLogger
import logging
import os
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from functools import partial
from collections import defaultdict
import sacrebleu
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
import evaluate

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

start_time = time.time()

logger = getLogger(__name__)
logsdir = 'logs'
logfile_name = 'spa_to_wayuu_tool_qwen7b_rl_sft_bleu.log'
logpath = os.path.join(logsdir, logfile_name)
if os.path.exists(logpath):
  os.remove(logpath)
logging.basicConfig(filename=logpath, encoding='utf-8', level=logging.DEBUG)
character = evaluate.load("character")

def get_policy_model(model_name):

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def calculator_tool(code_str, allowed_names=None):
    allowed_names = allowed_names or {}

    # Built-in safe functions you might allow
    safe_builtins = {
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum
    }

    try:
        result = ' <execution_result> ' + str(eval(code_str, {"__builtins__": safe_builtins, **allowed_names}, {})) + ' </execution_result>'
        logger.info(f'CORRECT USE OF CALCULATOR TOOL. Code: {code_str}, Result: {result}')
        return result
    except Exception as e:
        return f" <execution_result> Error: {e} </execution_result>"
    
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
        logger.debug(f'CORRECT USE OF SPA_TO_WAYU TOOL. Word: {spanish_word}, Result: {result}')
    else:
        result = " <matches> No matches found </matches>"
        logger.debug(f'NO_MATCHES SPA_TO_WAYU TOOL. Word: {spanish_word}')

    return result

TOOLS = [
    {
        'name': 'calculator',
        'description': 'A calculator tool that can perform basic arithmetic operations. You can use it to calculate the result of a mathematical expression.',
        'api': calculator_tool,
        'start_token': '<calculator>',
        'end_token': '</calculator>',
    },
    {
        'name': 'spa_to_wayu',
        'description': 'A tool that translates a word from Spanish to Wayuunaiki.',
        'api': spa_to_wayu_dictionary,
        'start_token': '<spa_to_wayuu>',
        'end_token': '</spa_to_wayuu>',
    }
]

def generate_batch_completion(model, tokenizer, prompts: list, return_ids=False, use_vllm=False, actions_num=1, **kwargs):
    batch = [[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ] for prompt in prompts]
    texts = tokenizer.apply_chat_template(
        batch,
        tokenize=False,
        add_generation_prompt=True,
    )

    default_sampling_args = {
        'max_new_tokens': 256,
        'temperature': 0.8,
        'top_p': 0.95,
    }
    default_sampling_args.update(kwargs)

    if use_vllm:
        model_inputs = tokenizer(texts)
    else:
        model_inputs = tokenizer(texts, padding='longest', padding_side='left', \
            return_tensors="pt")
    
    if use_vllm:
        inputs = model_inputs.input_ids
        dones = [False] * len(prompts)
        prompt_length = [len(input_ids) for input_ids in inputs]
        mask = [[1] * len(input_ids) for input_ids in inputs]
        responses = [""] * len(prompts)
        tools_enabled = kwargs.get('tools', [])
        stop_tokens = [tool['end_token'] for tool in tools_enabled] + [tokenizer.eos_token]
        for action_step in range(actions_num + 1 if len(tools_enabled) > 0 else 1):
            sampling_params = SamplingParams(temperature=default_sampling_args["temperature"], top_p=default_sampling_args['top_p'], top_k=-1, max_tokens=default_sampling_args['max_new_tokens'],
                stop=stop_tokens)
            outputs = model.generate(prompt_token_ids=inputs, sampling_params=sampling_params, lora_request=kwargs['lora_request'], use_tqdm=kwargs['use_tqdm'] if 'use_tqdm' in kwargs else None)

            for j, output in enumerate(outputs):
                if dones[j]:
                    continue
                
                for tool in tools_enabled:
                    if output.outputs[0].stop_reason == tool['end_token'] and tool['start_token'] in output.outputs[0].text:
                        api_args = output.outputs[0].text.split(tool['start_token'])[1].strip()
                        api_result = tool['api'](api_args)
                        responses[j] += output.outputs[0].text + f"{tool['end_token']}" + api_result
                        api_result_tokens = tokenizer.encode(api_result, return_tensors=None)
                        inputs[j] += list(output.outputs[0].token_ids) + api_result_tokens
                        mask[j] += [1] * len(output.outputs[0].token_ids) + [0] * len(api_result_tokens)
                        break # Only one tool can be used at a time
                if output.outputs[0].finish_reason == "stop" and output.outputs[0].stop_reason is None:
                    responses[j] += output.outputs[0].text
                    inputs[j] += list(output.outputs[0].token_ids)
                    dones[j] = True
                    mask[j] += [1] * len(output.outputs[0].token_ids)
                elif output.outputs[0].stop_reason not in stop_tokens:
                    logger.warning(f"Unexpected finish reason: {output.outputs[0].finish_reason} {output.outputs[0].stop_reason}")
                    responses[j] += tokenizer.eos_token
                    inputs[j] += [tokenizer.eos_token_id]
                    mask[j] += [1]
                    dones[j] = True

        if return_ids:
            generation_ids = inputs
            # padding the generation_ids to the max length
            max_length = max([len(ids) for ids in generation_ids])
            generation_ids = [ids + [tokenizer.pad_token_id]*(max_length-len(ids)) for ids in generation_ids]
            mask = [m + [0]*(max_length-len(m)) for m in mask]
            generation_ids = torch.tensor(generation_ids)
            return generation_ids, max(prompt_length), torch.tensor(mask)

        return responses
    
    if 'tools' in kwargs:
        raise NotImplementedError("Only vllm is supported for now when using tools")
        
    model_inputs = model_inputs.to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        **default_sampling_args
    )

    if return_ids:
        return generated_ids, len(model_inputs.input_ids[0])
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return response

import re

def extract_answer(response, transform_fn = lambda x: x, nan_val = None)->str|None:
    ans = re.match('.*?<answer>(.*?)</answer>\s*$', response, re.DOTALL|re.MULTILINE)
    if ans:
        try:
            return transform_fn(ans[1].strip())
        except:
            return nan_val
    return nan_val

import numpy as np
from tqdm import tqdm

def eval_multiplication(model, tokenizer, epochs=10, batch_size=128, generate_fn=generate_batch_completion):
    matches = 0
    tries = 0
    format_errors = 0
    for i in tqdm(range(epochs)):
        numbers = np.random.randint(0,101, (batch_size, 2))
        correct_result = numbers[:,0] * numbers[:,1]
        prompt = "What is the result of {} times {}?"
        prompts = [prompt_template.format(prompt.format(*nums)) for nums in numbers]
        responses = generate_fn(model, tokenizer, prompts)
        answer = np.array([extract_answer(response, lambda x: int(x) if x.isnumeric() else x) for response in responses])
        format_errors += (answer == None).sum()
        matches += (correct_result == answer).sum()
        tries += len(correct_result)

        del responses
        del answer
        torch.cuda.empty_cache()
    
    acc = matches/tries
    format_errors /= tries
    wrong_answer = 1 - acc - format_errors
    return {
        'acc': acc,
        'format_errors': format_errors,
        'wrong_answer': wrong_answer
    }

def eval_translations(model, tokenizer, dataset, prompt_template, batches=10, batch_size=128, generate_fn=generate_batch_completion):
    bleu_sum = 0
    samples_num = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i in tqdm(range(batches)):
        prompts, answers = next(iter(dataloader))
        prompts = [prompt_template.format(prompt) for prompt in prompts]
        responses = generate_fn(model, tokenizer, prompts, use_tqdm=False, temperature=0, top_p=1, max_new_tokens=768)
        samples = [extract_answer(response, nan_val="") for response in responses]

        bleu = sacrebleu.BLEU(effective_order = True)
        def get_bleu_score(sample, correct_translation):
            # Compute bleu score for each sample. 
            # Bleu score normalized to [0, 1]
            return bleu.sentence_score(sample, 
                                    [correct_translation]
                                    ).score

        answer_bleu_scores = [
            get_bleu_score(sample, answer)
            for sample, answer in zip(samples, answers)
        ]
        bleu_sum += sum(answer_bleu_scores)
        samples_num += len(samples)

        del responses
        del answer_bleu_scores
        torch.cuda.empty_cache()
    
    return bleu_sum/samples_num

import torch
from torch import nn


# %%
def make_rollouts(model, simulations, initial_prompt: str, max_size = 256, temperature=1.0, **kwargs):
    prompts = [initial_prompt]*simulations
    
    with torch.no_grad():
        generations, prompt_length, mask = generate_batch_completion(model, tokenizer, prompts, return_ids=True, temperature=temperature, top_p=1, max_new_tokens=max_size, **kwargs)

    # Create mask for padding and eos tokens
    is_terminal = torch.zeros_like(generations, device='cpu')
    is_terminal[generations == tokenizer.pad_token_id] = 1
    eos_token = is_terminal.shape[1]-is_terminal.count_nonzero(dim=1)-1
    is_terminal[torch.arange(len(is_terminal)), eos_token] = 1
    return generations[:,prompt_length:], is_terminal[:,prompt_length:], generations, prompt_length, mask[:,prompt_length:]

# %%
def get_rewards(samples, is_terminal, correct_result):
    samples = samples.cpu()
    is_terminal = is_terminal.cpu()
    rewards = torch.zeros_like(samples, dtype=torch.float)
    # Extract the answer from the answer tag if any on each response
    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    logger.debug(f'samples: {samples}')
    answer = torch.tensor([extract_answer(response, lambda x: int(x) if x.isnumeric() else correct_result+1, torch.nan) for response in samples])

    eos_index = (is_terminal == 0).sum(dim=1)
    logger.info(f'Response length mean: {eos_index.to(torch.float16).mean():,.2f}')
    eos_index = torch.min(eos_index, torch.tensor(is_terminal.shape[1]-1))

    answer_is_correct = (answer == correct_result)
    answer_is_not_correct = (answer != correct_result)
    wrong_format = answer.isnan()
    answer_is_correct_count = answer_is_correct.sum()
    answer_is_not_correct_count = answer_is_not_correct.sum()
    wrong_format_count = wrong_format.sum()
    logger.debug(f'Correct answer: {correct_result} Extracted: {answer}')
    logger.debug(f'Correct: {answer_is_correct_count}, Wrong_format: {wrong_format_count}, Wrong_anser: {answer_is_not_correct_count-wrong_format_count}')

    # 0.5 reward point if the response has an answer tag
    # rewards[torch.arange(len(samples)), eos_index] = (1-wrong_format.to(torch.float32))*0.5
    # An additional 1 point of reward if the answer is correct
    rewards[torch.arange(len(samples)), eos_index] += answer_is_correct.to(torch.float32)
    logger.debug(f'Rewards: {rewards[torch.arange(len(samples)), eos_index]}')
    return rewards

def get_rewards_translation(samples, is_terminal, correct_translation):
    samples = samples.cpu()
    is_terminal = is_terminal.cpu()
    rewards = torch.zeros_like(samples, dtype=torch.float)

    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    logger.debug(f'samples: {samples}')
    samples = [extract_answer(response, nan_val="") for response in samples]

    bleu = sacrebleu.BLEU(effective_order = True)
    def get_bleu_score(sample, correct_translation):
        # Compute bleu score for each sample. 
        # Bleu score normalized to [0, 1]
        return bleu.sentence_score(sample, 
                                   [correct_translation]
                                   ).score / 100.0 

    answer_bleu_scores = torch.tensor([
        get_bleu_score(sample, correct_translation)
        for sample in samples
    ])

    eos_index = (is_terminal == 0).sum(dim=1)
    eos_index = torch.min(eos_index, torch.tensor(is_terminal.shape[1]-1))

    # Assign rewards based in BLEU score
    rewards[torch.arange(len(samples)), eos_index] += answer_bleu_scores
    logger.debug(f'Rewards: {rewards[torch.arange(len(samples)), eos_index]}')
    
    return rewards

def get_rewards_translation_character(samples, is_terminal, correct_translation):
    samples = samples.cpu()
    is_terminal = is_terminal.cpu()
    rewards = torch.zeros_like(samples, dtype=torch.float)

    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    logger.debug(f'samples: {samples}')
    samples = [extract_answer(response, nan_val="") for response in samples]

    
    def get_character_score(sample, correct_translation):
        # Compute character score for each sample. 
        # Character score between 0 and 1
        score = character.compute(
            references=[correct_translation], predictions=[sample]
            )["cer_score"]
        return 1 - score

    answer_character_scores = torch.tensor([
        get_character_score(sample, correct_translation)
        for sample in samples
    ])

    eos_index = (is_terminal == 0).sum(dim=1)
    eos_index = torch.min(eos_index, torch.tensor(is_terminal.shape[1]-1))

    # Assign rewards based on character score
    rewards[torch.arange(len(samples)), eos_index] += answer_character_scores
    logger.debug(f'Rewards: {rewards[torch.arange(len(samples)), eos_index]}')
    
    return rewards

prompt_template = """Answer the given question. You must conduct reasoning inside <think> and </think>.
After reasoning, if you find you need to perform a calculation, you can call a calculator tool by passing a python expression inside
the calculator tags, in this way: <calculator> expression </calculator>, and it will return the of the calculation
between <execution_result> and </execution_result>. You can use the calculator as many times as you want.
If you find no further calculation is needed, you can directly provide the answer inside <answer> and </answer> without
detailed illustrations. For example, <answer> xxx </answer>. Question: {}.
"""

def run_one_mul_simulation(model, generations_num, temperature=1.0, **kwargs):
    # Generate the random numbers to be multiplied to create the prompt
    numbers = torch.randint(0,101, (1, 2))
    # numbers = torch.tensor([[23, 37]])
    correct_result = numbers[:,0] * numbers[:,1]
    prompt = "What is the result of {} times {}?"
    prompts = [prompt_template.format(prompt.format(*nums)) for nums in numbers]

    # Generate the responses for the prompt
    inputs, is_terminal, complete_prompts, prompt_length, mask = make_rollouts(model, generations_num, prompts[0], temperature=temperature, **kwargs)
    # Calculate the rewards for each response
    rewards = get_rewards(inputs, is_terminal, correct_result)
    return inputs, rewards, is_terminal, complete_prompts, prompt_length, mask


translate_prompt_template_tool="""Translate the following Spanish text into Wayuunaiki.
Begin by reasoning between <think> and </think>, identifying any words or phrases you're unsure how to translate.
If needed, you may look up words using the dictionary tool by wrapping the Spanish word in <spa_to_wayuu> and </spa_to_wayuu>. The dictionary will return matches enclosed in <matches> and </matches>. You can use the dictionary as many times as necessary.
Once you have all the information you need, provide the final translation enclosed in <answer> and </answer>. For example: <answer> xxx </answer>.

Spanish text: {}"""

translate_prompt_template = """Translate the following Spanish text to Wayuunaiki.
Provide the translated text inside <answer> and </answer>. For example, <answer> xxx </answer>. Text in spanish: {}"""

def translation_simulation(model, generations_num, temperature=1.0, **kwargs):
    spanish_text = kwargs['spa']
    wayuu_text = kwargs['wayuu']
    logger.debug(f'Texto en español: {spanish_text}')
    logger.debug(f'Traducción Wayuu: {wayuu_text}')
    prompt = translate_prompt_template_tool.format(spanish_text)

    # Generate the responses for the prompt
    if 'max_new_tokens' in kwargs:
        max_size = kwargs['max_new_tokens']
        del kwargs['max_new_tokens']
    else:
        max_size = None
    inputs, is_terminal, complete_prompts, prompt_length, mask = make_rollouts(model, generations_num, prompt, temperature=temperature, max_size=max_size, **kwargs)
    # Calculate the rewards for each response
    rewards = get_rewards_translation(inputs, is_terminal, wayuu_text)
    # rewards = get_rewards_translation_character(inputs, is_terminal, wayuu_text)
    return inputs, rewards, is_terminal, complete_prompts, prompt_length, mask


# %%
def compute_advantages(rewards, is_terminal, gamma=1.0, gae_lambda=0.2, dr_grpo=False):
    # Find the longest response in the batch
    num_rollout_steps = torch.max((is_terminal==0).sum(1))
    num_rollout_steps = torch.min(num_rollout_steps, torch.tensor(rewards.shape[1])-1)
    advantages = torch.zeros((len(rewards), torch.tensor(rewards.shape[1])))

    eos_index = (is_terminal == 0).sum(dim=1)
    eos_index = torch.min(num_rollout_steps, eos_index)
    rewards_of_outputs = rewards[torch.arange(len(rewards)), eos_index]
    norm_rewards = (rewards_of_outputs - rewards_of_outputs.mean()) 
    # if not dr_grpo:
    #     norm_rewards /= (rewards_of_outputs.std() + 1e-8)
    norm_rewards /= (rewards_of_outputs.std() + 1e-8)

    for i in range(len(rewards)):
        advantages[i,:eos_index[i]+1] = norm_rewards[i]

    logger.debug(f'advantages: {advantages}')
    return advantages

# %%
import numpy as np


# %%
def update_policy(model, ref_model, old_model, optimizer, is_terminal, advantanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=None, normalize_advantage=False, lower_clip=None, upper_clip=None, kl_penalty_coef=0.04, dr_grpo=False, no_kl=False, temperature=1.0, use_deepspeed=False, mask=None):
    lower_clipped_threshold = lower_clip
    upper_clipped_threshold = upper_clip


    is_terminal = is_terminal.to(model.device)
    advantanges = advantanges.to(model.device)
    complete_prompts = complete_prompts.to(model.device)
    if mask is not None:
        mask = mask.to(model.device)
        # Mask the advantages
        advantanges = advantanges * mask
    for epoch in range(update_epochs):
        batch_indices = np.arange(len(advantanges))
        np.random.shuffle(batch_indices)
        minibatches= len(advantanges) // minibatch_size
        batch_size = len(batch_indices)

        ## Iterate in minibatches (random minibatch_size responses)
        for start in range(0, len(advantanges), minibatch_size):
            end = start + minibatch_size
            minibatch_indices = batch_indices[start:end]
            # Calculate the logits for the generated responses with the policy model (the logits of the previous token represents the distribution of the current token, that's why the -1)
            logits = model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
            logits /= temperature
            if old_model is not None:
                with torch.no_grad():
                    old_logits  = old_model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
                    old_logits /= temperature
            # Calculate the logits for the generated responses with the ref model
            if ref_model is not None:
                with torch.no_grad():
                    ref_logits  = ref_model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
                    ref_logits /= temperature
            else:
                with torch.no_grad():
                    if use_deepspeed:
                        with model.module.disable_adapter():
                            ref_logits  = model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
                            ref_logits /= temperature
                    else:
                        with model.disable_adapter():
                            ref_logits  = model(complete_prompts[minibatch_indices,:prompt_length+advantanges.shape[1]]).logits[:,prompt_length-1:-1]
                            ref_logits /= temperature
            # Get the ids of the actual generated tokens
            completion_ids = generations[minibatch_indices]
            completion_ids = generations[minibatch_indices,:advantanges.shape[1]].reshape(-1)
            actual_minibatch_size = len(logits)
            max_tokens = advantanges.shape[1]
            # Calculate the probabilities of each token generated with the policy and ref model
            probs = nn.functional.softmax(logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
            probs_tokens = probs[torch.arange(len(completion_ids)), completion_ids].reshape(advantanges.shape)
            log_probs_sum = torch.log(probs_tokens)
            if old_model:
                old_probs = nn.functional.softmax(old_logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
                old_probs_tokens = old_probs[torch.arange(len(completion_ids)), completion_ids]
                log_old_probs_sum = torch.log(old_probs_tokens).reshape(advantanges.shape)
            else:
                old_probs_tokens = probs_tokens.detach().reshape(advantanges.shape)
                log_old_probs_sum = log_probs_sum.detach()
            ref_probs = nn.functional.softmax(ref_logits.reshape(actual_minibatch_size*max_tokens,-1), dim=1)
            ref_probs_tokens = ref_probs[torch.arange(len(completion_ids)), completion_ids]
            log_ref_probs_sum = torch.log(ref_probs_tokens).reshape(advantanges.shape)

            # terminal state is shift to the right so the eos token that has the reward is taken in account
            minibatch_is_not_terminal = torch.cat((torch.ones(actual_minibatch_size,1, device=model.device), 1-is_terminal[minibatch_indices]), dim=1)[:,:advantanges.shape[1]]

            # Track KL divergence
            log_prob_ratio = log_probs_sum - log_ref_probs_sum
            probability_ratio = log_prob_ratio.exp()
            minibatch_approx_kl = ((probability_ratio - 1) - log_prob_ratio) * minibatch_is_not_terminal
            minibatch_approx_kl_by_generation = (minibatch_approx_kl.sum(dim=1) / minibatch_is_not_terminal.count_nonzero(dim=1))
            minibatch_approx_kl_mean = minibatch_approx_kl_by_generation.mean()
            logger.info(f'minibatch_approx_kl_by_generation: {minibatch_approx_kl_by_generation}')
            logger.debug(f'Approx KL divergence of minibatch: {minibatch_approx_kl_mean:.6f}')

            minibatch_advantages = advantanges[minibatch_indices,:advantanges.shape[1]] * minibatch_is_not_terminal

            # The policy loss is to maximize the probability_ratio times the advantages
            new_old_prob_ratio = (log_probs_sum-log_old_probs_sum).exp()
            # Verification in case the old model is the same as the current one
            logger.debug(f'new_old_prob_ratio. This should be 1, {(new_old_prob_ratio*minibatch_is_not_terminal).sum()/minibatch_is_not_terminal.count_nonzero()}')
            loss = new_old_prob_ratio * minibatch_advantages
            logger.debug(f'probability ratio: mean - {probability_ratio.mean()}, min - {probability_ratio.min()}, max: {probability_ratio.max()}')
            # Clipped loss: Only considers the probability_ratio change between a reasonable range
            if lower_clipped_threshold != None and upper_clipped_threshold != None:
                clipped_loss = torch.clamp(new_old_prob_ratio, lower_clipped_threshold, upper_clipped_threshold) * minibatch_advantages
            else:
                clipped_loss = loss

            logger.debug(f'generation_length: {minibatch_is_not_terminal.count_nonzero(dim=1)}')
            # Take the min to be pessimistic
            if dr_grpo:
                loss_mean = torch.min(loss, clipped_loss).sum(dim=1).mean()
            else:
                loss_avg_by_generation = torch.min(loss, clipped_loss).sum(dim=1) / minibatch_is_not_terminal.count_nonzero(dim=1)
                logger.debug(f'loss_avg_by_generation: {loss_avg_by_generation}')
                loss_mean = loss_avg_by_generation.mean()
                
            logger.debug(f'loss_mean: {loss_mean}')
            # Add the minus to perform optimization minimizing the loss
            if dr_grpo or no_kl:
                loss = -loss_mean
            else:
                loss_with_kl_penalty = loss_mean - kl_penalty_coef*minibatch_approx_kl_mean
                loss = -loss_with_kl_penalty

            logger.debug(f'loss: {loss.item()}')

            # # Update the policy weights
            # if not use_deepspeed:
            #     optimizer.zero_grad()
            #     loss.backward()
            #     nn.utils.clip_grad_norm_(model.parameters(), 0.1) # Avoid large gradients
            #     optimizer.step()
            # else:
            #     model.backward(loss, scale_wrt_gas=False)
            #     model.step()

            # Update the scheduler every rl step no matter the epochs
            if scheduler:
                scheduler.step()
            return loss

def update_vllm_instance(vllm_instance, model, just_validate=False)->bool:
    adapter_id = 1

    fused_layers_mapping = {
        'gate_up_proj': ['gate_proj', 'up_proj'],
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
    }
    fused_layers_mapping = defaultdict(lambda: [None], fused_layers_mapping)

    lora_layers = vllm_instance.llm_engine.model_executor.driver_worker.model_runner.lora_manager._adapter_manager.list_adapters()[adapter_id].loras
    policy_model_state_dict = model.base_model.model.state_dict()
    adapater_name = 'default'
    for lora_layer_name in lora_layers:
        layer_type = lora_layer_name.split('.')[-1]
        usfused_layers_names = fused_layers_mapping[layer_type]

        for i, unfused_layer_name in enumerate(usfused_layers_names):
            lora_a_name = lora_layer_name + f'.lora_A.{adapater_name}.weight'
            lora_b_name = lora_layer_name + f'.lora_B.{adapater_name}.weight'

            # Fix name deviation
            if unfused_layer_name is not None:
                lora_a_name = lora_a_name.replace(layer_type, unfused_layer_name)
                lora_b_name = lora_b_name.replace(layer_type, unfused_layer_name)

            if lora_a_name not in policy_model_state_dict or lora_b_name not in policy_model_state_dict:
                logger.warning(f"Layer: {lora_layer_name} not found in state dict. Lora A: {lora_a_name}, Lora B: {lora_b_name}")
                continue
            vllm_layer = lora_layers[lora_layer_name]

            hf_layer_lora_a = policy_model_state_dict[lora_a_name]
            hf_layer_lora_b = policy_model_state_dict[lora_b_name]

            vllm_device = 'cpu'
            if just_validate:
                if isinstance(vllm_layer.lora_a, list):
                    lora_a_is_equal = torch.equal(hf_layer_lora_a.T.to(torch.bfloat16).to(vllm_device), vllm_layer.lora_a[i])
                    lora_b_is_equal = torch.equal(hf_layer_lora_b.T.to(torch.bfloat16).to(vllm_device), vllm_layer.lora_b[i])
                else:
                    lora_a_is_equal = torch.equal(hf_layer_lora_a.T.to(torch.bfloat16).to(vllm_device), vllm_layer.lora_a)
                    lora_b_is_equal = torch.equal(hf_layer_lora_b.T.to(torch.bfloat16).to(vllm_device), vllm_layer.lora_b)

                assert lora_a_is_equal, f"LoRA A weights do not match for {lora_layer_name} {unfused_layer_name}"
                assert lora_b_is_equal, f"LoRA B weights do not match for {lora_layer_name} {unfused_layer_name}"
            else:
                if isinstance(vllm_layer.lora_a, list):
                    vllm_layer.lora_a[i] = hf_layer_lora_a.T.to(torch.bfloat16).to(vllm_device)
                    vllm_layer.lora_b[i] = hf_layer_lora_b.T.to(torch.bfloat16).to(vllm_device)
                else:
                    vllm_layer.lora_a = hf_layer_lora_a.T.to(torch.bfloat16).to(vllm_device)
                    vllm_layer.lora_b = hf_layer_lora_b.T.to(torch.bfloat16).to(vllm_device)

                # Activate the new weights
                vllm_instance.llm_engine.model_executor.driver_worker.model_runner.lora_manager._adapter_manager.deactivate_adapter(adapter_id)
                assert adapter_id not in vllm_instance.llm_engine.model_executor.driver_worker.model_runner.lora_manager._adapter_manager._active_adapters, f"Adapter {adapter_id} was not deactivated!"
                # Update the LoRA weights again for the forward pass
                vllm_instance.llm_engine.model_executor.driver_worker.model_runner.lora_manager._adapter_manager.activate_adapter(adapter_id)

    return True


# Cosine scheduler with warmup
from torch.optim.lr_scheduler import CosineAnnealingLR
class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, warmup_start_lr=0.0, eta_min=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            warmup_factor = self.current_epoch / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.warmup_start_lr + warmup_factor * (self.base_lrs[i] - self.warmup_start_lr)
        else:
            self.cosine_scheduler.step()
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

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
# %% [markdown]
# ## Putting everything together

# %%
from peft import LoraConfig, get_peft_model 

# HYPERPARAMETERS

# %%
max_steps = 400*8
sims_per_prompt = 8
rl_steps = max_steps // sims_per_prompt
minibatch_size = 8
update_epochs = 1
policy_lr = 5e-6
kl_penalty_coef = 0.04
warmup_steps = 25
use_vllm = True # Use vllm for inference
save_adapter_path = 'models/tools_sft_rl_qwen7b_model'
best_adapter_path = 'models/tools_sft_rl_qwen7b_best_model'
base_model_name = "Qwen/Qwen2.5-7B-Instruct"
spanish_train_file = 'datasets/train.es.txt'
wayuu_train_file = 'datasets/train.guc.txt'
spanish_val_file = 'datasets/dev.es.txt'
wayuu_val_file = 'datasets/dev.guc.txt'
use_deepspeed = False # Use deepspeed for training
update_ref_model_steps = None
gae_lambda = 1.0
normalize_advantage=False
temperature=1.0 # Temperature for the generations
lower_clip=0.8
upper_clip=1.2
dr_grpo = True
no_kl=True
max_new_tokens=320
enabled_tools = ['spa_to_wayu']
checkpoint_to_start = "models/sft_base_qwen7b_tools"
# checkpoint_to_start = None
action_calls = 4
accum_grad_steps = 8
logger.info(f'Hyperparameters:\nupdate_epochs:{update_epochs}\nrl_steps:{rl_steps}\nsims_per_prompt:{sims_per_prompt}\nminibatch_size:{minibatch_size}\npolicy_lr:{policy_lr}\nwarmup_steps:{warmup_steps}\ngae_lambda: {gae_lambda}\nnormalize advantage:{normalize_advantage}\nlower_clip:{lower_clip}\nupper_clip:{upper_clip}\nkl_penalty_coef:{kl_penalty_coef}\ntemperature:{temperature}\ndr_grpo:{dr_grpo}\nno_kl={no_kl}\nuse_deepspeed={use_deepspeed}\nuse_vllm={use_vllm}\nenabled_tools={enabled_tools}\nbase_model_name={base_model_name}\nspanish_train_file={spanish_train_file}\nwayuu_train_file={wayuu_train_file}\nmax_new_tokens={max_new_tokens}\ncheckpoint_to_start={checkpoint_to_start}\naccum_grad_steps={accum_grad_steps}\naction_calls={action_calls}\nspanish_val_file={spanish_val_file}\nwayuu_val_file={wayuu_val_file}\nbest_adapter_path={best_adapter_path}\nsave_adapter_path={save_adapter_path}')

tools = [tool for tool in TOOLS if tool['name'] in enabled_tools]
generate_batch_completion = partial(generate_batch_completion, tools=tools, actions_num=action_calls)

model, tokenizer = get_policy_model(base_model_name)
# ref_model, _ = get_policy_model()
# ref_model.eval()

config = LoraConfig(
    r=64, # Rank de las matrices A y B
    lora_alpha=64, # Factor de regularización de las matrices A y B
    target_modules= [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # lora_dropout=0.05, # Dropout de las matrices A y B
    # bias="none", # No se añade bias a las capas lineales
    task_type="CAUSAL_LM" # Tipo de tarea
)

# Use LoRA to finetune the policy model
if checkpoint_to_start:
    logger.info(f'Loading checkpoint {checkpoint_to_start}')
    model = PeftModel.from_pretrained(model, checkpoint_to_start, is_trainable=True)
else:
    model = get_peft_model(model, config)

optimizer = torch.optim.AdamW(model.parameters(), lr=policy_lr, betas=(0.9, 0.99), weight_decay=0.1) if not use_deepspeed else None
# scheduler = CosineAnnealingLR(optimizer, T_max=rl_steps)
scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, rl_steps) if not use_deepspeed else None
if not use_deepspeed:
    scheduler.step()

if use_deepspeed:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    import os
    import socket

    def find_free_port():
        """Find a free port on localhost."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    # Needed to stop DeepSpeed from complaining
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_port())
    os.environ["RANK"] = "1"
    os.environ["LOCAL_RANK"] = "1"
    os.environ["WORLD_SIZE"] = "1"
    deepspeed_config = {
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2, "overlap_comm": False},
            "train_batch_size": sims_per_prompt*accum_grad_steps,
            "train_micro_batch_size_per_gpu": minibatch_size,
            "gradient_accumulation_steps": accum_grad_steps,
            "gradient_clipping": 0.1,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr":policy_lr,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "torch_adam": True,
                },
            },
        }
    ref_deepspeed_config = {
            "bf16": {"enabled": True},
            "train_batch_size": sims_per_prompt,
            "train_micro_batch_size_per_gpu": minibatch_size,
            "gradient_accumulation_steps": 1,
        }
    model_engine, *_ = deepspeed.initialize(
        model=model,
        config=deepspeed_config,
        model_parameters=model.parameters(),
    )
else:
    # model_engine = torch.compile(model)
    model_engine = model

if use_vllm:
    model.save_pretrained(save_adapter_path)

    inference_engine = LLM(
    model=base_model_name,
    enable_lora=True,
    max_lora_rank=64,
    max_loras=1,
    gpu_memory_utilization=0.4,
    # enable_prefix_caching=True,
    swap_space=6,
    scheduling_policy="fcfs",
    dtype=torch.bfloat16,
    max_model_len=464,
    # enable_sleep_mode=True,
    device="cuda:2"
    )

    # Load the LoRA adapter
    lora_request = LoRARequest('tuning', 1, lora_path=save_adapter_path)
    inference_engine.generate(['Hello'], SamplingParams(max_tokens=2), lora_request=lora_request)

    # Update the LoRA weights to match the policy model
    update_vllm_instance(inference_engine, model_engine)
    assert update_vllm_instance(inference_engine, model_engine, just_validate=True)

ref_model = None

# Load the dataset
dataset = TextDataset(spanish_train_file, wayuu_train_file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load validation dataset
validation_dataset = TextDataset(spanish_train_file, wayuu_train_file)

import copy
# Training loop
try:
    model_engine.eval()
    if use_vllm:
        acc = 0
        # acc = eval_translations(inference_engine, tokenizer, validation_dataset, translate_prompt_template_tool, batches=10, batch_size=64, generate_fn=partial(generate_batch_completion, use_vllm=True, lora_request=lora_request, max_new_tokens=max_new_tokens))
    else:
        raise ValueError('use_vllm is False')
    logger.info(f'Evaluation before training: {acc}')

    max_performance = acc

    model_engine.train()
    old_model = None

    rl_step = 0
    accumulated_grad_steps = 0
    while rl_step < rl_steps:
        logger.info(f'rl_step: {rl_step+1:,}')
        spa_sample, wayuu_sample = next(iter(dataloader))
        spa_sample, wayuu_sample = spa_sample[0], wayuu_sample[0]
        # spa_sample, wayuu_sample = dataset[0]
        if use_vllm:
            # generations, rewards, is_terminal, complete_prompts, prompt_length, mask = run_one_mul_simulation(inference_engine, sims_per_prompt, temperature=temperature, use_vllm=use_vllm, lora_request=LoRARequest('tuning', 1, lora_path=save_adapter_path))
            generations, rewards, is_terminal, complete_prompts, prompt_length, mask = translation_simulation(inference_engine, sims_per_prompt, temperature=temperature, use_vllm=use_vllm, lora_request=LoRARequest('tuning', 1, lora_path=save_adapter_path), spa=spa_sample, wayuu=wayuu_sample, max_new_tokens=max_new_tokens)
        else:
            # generations, rewards, is_terminal, complete_prompts, prompt_length = run_one_mul_simulation(model_engine, sims_per_prompt, temperature=temperature)
            generations, rewards, is_terminal, complete_prompts, prompt_length = translation_simulation(model_engine, sims_per_prompt, temperature=temperature, spa=spa_sample, wayuu=wayuu_sample, max_new_tokens=max_new_tokens)
        advantanges = compute_advantages(rewards, is_terminal, gae_lambda=gae_lambda, dr_grpo=dr_grpo)

        if generations.shape[1] > 320:
            logger.warning(f'Generations shape is too large: {generations.shape}. Skipping this step.')
            continue
        if (advantanges == 0).all().item():
            continue

        logger.info('Updating policy')
        if scheduler:
            logger.debug(f'Learning rate: {scheduler.get_lr()}')
        loss = update_policy(model_engine, ref_model, old_model, optimizer, is_terminal, advantanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=scheduler, normalize_advantage=normalize_advantage, lower_clip=lower_clip, upper_clip=upper_clip, dr_grpo=dr_grpo, no_kl=no_kl, temperature=temperature, use_deepspeed=use_deepspeed, mask=mask)
        loss = loss / accum_grad_steps
        accumulated_grad_steps += 1
        if not use_deepspeed:
            loss.backward()
        else:
            model_engine.backward(loss, scale_wrt_gas=False)

        if use_deepspeed:
            model_engine.step()
        elif accumulated_grad_steps == accum_grad_steps:
            # Update the policy weights
            nn.utils.clip_grad_norm_(model_engine.parameters(), 1.0) # Avoid large gradients
            optimizer.step()
            optimizer.zero_grad()
            
            accumulated_grad_steps = 0


        if use_vllm:
            # Update the LoRA adapter
            update_vllm_instance(inference_engine, model_engine)

        # Track progress on specific task
        if (rl_step+1)%50 == 0:
            model_engine.eval()
            with torch.no_grad():
                if use_vllm:
                    # update_vllm_instance(inference_engine, model_engine)
                    acc = eval_translations(inference_engine, tokenizer, validation_dataset, translate_prompt_template_tool, batches=10, batch_size=64, generate_fn=partial(generate_batch_completion, use_vllm=True, lora_request=lora_request))
                else:
                    raise ValueError('use_vllm is False')
            logger.info(f'Evaluation on rl step {rl_step+1:,}: {acc}')
            model_engine.train()

            # Save the model if the performance is better
            if acc > max_performance:
                max_performance = acc
                logger.info(f'Saving model with performance {max_performance}')
                model_engine.save_pretrained(best_adapter_path)
                logger.info(f'Best model saved to {best_adapter_path}')


        if update_ref_model_steps is not None and (rl_step+1)%update_ref_model_steps == 0:
            ref_model = copy.deepcopy(model_engine).eval() # Update the ref model

        rl_step += 1

except Exception as error:
    logger.critical(error, exc_info=True)
    pass

model_engine.eval()
with torch.no_grad():
    if use_vllm:
        acc = eval_translations(inference_engine, tokenizer, validation_dataset, translate_prompt_template_tool, batches=10, batch_size=64, generate_fn=partial(generate_batch_completion, use_vllm=True, lora_request=lora_request))
    else:
        raise ValueError('use_vllm is False')
logger.info(f'Evaluation after training: {acc}')
model_engine.save_pretrained(save_adapter_path)


end_time = time.time()
execution_time = end_time - start_time
logger.info(f'Total execution time in minutes: {execution_time/60:.2f}')