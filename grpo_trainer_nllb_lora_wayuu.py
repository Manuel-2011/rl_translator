# %%
import os
import gc
import re
import copy

import torch
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from logging import getLogger
import logging
import os
import time
from functools import partial
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import evaluate
import sacrebleu
from transformers.tokenization_utils import AddedToken


start_time = time.time()

logger = getLogger(__name__)
logsdir = 'logs'
logfile_name = 'grpo-DRGrpo1-vllm-no-partial-reward-NLLB.log'
logpath = os.path.join(logsdir, logfile_name)
if os.path.exists(logpath):
  os.remove(logpath)
logging.basicConfig(filename=logpath, encoding='utf-8', level=logging.DEBUG)
character = evaluate.load("character")

def get_policy_model(model_name, src_lang, tgt_lang):

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)#, tgt_lang=tgt_lang)
    tokenizer.add_tokens(AddedToken(tgt_lang, normalized=False, special=True))
    return model, tokenizer

def generate_batch_completion(model, tokenizer, prompts: list, return_ids=False, **kwargs):
    default_sampling_args = {
        'do_sample': True, # FIXME not enough memory in local
        'max_new_tokens': 256,
        'temperature': 0.8,
        'top_p': 0.35, # FIXME not enough memory in local
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

def eval_translations(model, tokenizer, dataset, batches=10, batch_size=128, generate_fn=generate_batch_completion):
    bleu_sum = 0
    samples_num = 0
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for i in tqdm(range(batches)):
        prompts, answers = next(iter(dataloader))
        prompts = [prompt for prompt in prompts]
        responses = generate_fn(model, tokenizer, prompts)

        bleu = sacrebleu.BLEU(effective_order = True)
        def get_bleu_score(sample, correct_translation):
            # Compute bleu score for each sample. 
            # Bleu score normalized to [0, 1]
            return bleu.sentence_score(sample, 
                                    [correct_translation]
                                    ).score

        answer_bleu_scores = [
            get_bleu_score(sample, answer)
            for sample, answer in zip(responses, answers)
        ]
        bleu_sum += sum(answer_bleu_scores)
        samples_num += len(responses)

        del responses
        del answer_bleu_scores
        torch.cuda.empty_cache()

    return bleu_sum/samples_num

def make_rollouts(model, simulations, initial_prompt: str, max_size = 256, temperature=1.0, **kwargs):
    prompts = [initial_prompt]*simulations
    
    with torch.no_grad():
        model_params = {k: v for k, v in kwargs.items() if k not in ["spa", "wayuu"]}
        generations, prompt_length = generate_batch_completion(model, tokenizer, prompts, return_ids=True, temperature=temperature, top_p=1, max_new_tokens=max_size, **model_params)

    # Create mask for padding and eos tokens
    is_terminal = torch.zeros_like(generations, device='cpu')
    is_terminal[generations == tokenizer.pad_token_id] = 1
    eos_token = is_terminal.shape[1]-is_terminal.count_nonzero(dim=1)-1
    is_terminal[torch.arange(len(is_terminal)), eos_token] = 1
    return generations[:,prompt_length:], is_terminal[:,prompt_length:], generations, prompt_length

def get_rewards_translation_character(samples, is_terminal, correct_translation):
    samples = samples.cpu()
    is_terminal = is_terminal.cpu()
    rewards = torch.zeros_like(samples, dtype=torch.float)

    samples = tokenizer.batch_decode(samples, skip_special_tokens=True)
    logger.debug(f'samples: {samples}')
    
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

def translation_simulation(model, generations_num, temperature=1.0, **kwargs):
    spanish_text = kwargs['spa']
    wayuu_text = kwargs['wayuu']
    logger.debug(f'Texto en español: {spanish_text}')
    logger.debug(f'Traducción Wayuu: {wayuu_text}')

    # Generate the responses for the prompt
    if 'max_new_tokens' in kwargs:
        max_size = kwargs['max_new_tokens']
        del kwargs['max_new_tokens']
    else:
        max_size = None
    inputs, is_terminal, complete_prompts, prompt_length = make_rollouts(model, generations_num, spanish_text, temperature=temperature, max_size=max_size, **kwargs)
    # Calculate the rewards for each response
    # rewards = get_rewards_translation(inputs, is_terminal, wayuu_text)
    rewards = get_rewards_translation_character(inputs, is_terminal, wayuu_text)
    return inputs, rewards, is_terminal, complete_prompts, prompt_length


def compute_advantages(rewards, is_terminal, gamma=1.0, gae_lambda=0.2, dr_grpo=False):
    # Find the longest response in the batch
    num_rollout_steps = torch.max((is_terminal==0).sum(1))
    num_rollout_steps = torch.min(num_rollout_steps, torch.tensor(rewards.shape[1])-1)
    advantages = torch.zeros((len(rewards), torch.tensor(rewards.shape[1])))

    eos_index = (is_terminal == 0).sum(dim=1)
    eos_index = torch.min(num_rollout_steps, eos_index)
    rewards_of_outputs = rewards[torch.arange(len(rewards)), eos_index]
    norm_rewards = (rewards_of_outputs - rewards_of_outputs.mean()) 
    if not dr_grpo:
        norm_rewards /= (rewards_of_outputs.std() + 1e-8)

    for i in range(len(rewards)):
        advantages[i,:eos_index[i]+1] = norm_rewards[i]

    logger.debug(f'advantages: {advantages}')
    return advantages


def update_policy(model, ref_model, old_model, optimizer, is_terminal, advantanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=None, normalize_advantage=False, lower_clip=None, upper_clip=None, kl_penalty_coef=0.04, dr_grpo=False, no_kl=False, temperature=1.0):
    lower_clipped_threshold = lower_clip
    upper_clipped_threshold = upper_clip


    is_terminal = is_terminal.to(model.device)
    advantanges = advantanges.to(model.device)
    complete_prompts = complete_prompts.to(model.device)
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
            logits = model(
                input_ids=complete_prompts.to(model.device)[:,:prompt_length],
                decoder_input_ids=complete_prompts.to(model.device)[:,prompt_length:]
            ).logits
            logits /= temperature
            if old_model is not None:
                with torch.no_grad():
                    old_logits = old_model(
                        input_ids=complete_prompts.to(old_model.device)[:,:prompt_length],
                        decoder_input_ids=complete_prompts.to(old_model.device)[:,prompt_length:]
                    ).logits
                    old_logits /= temperature
            # Calculate the logits for the generated responses with the ref model
            if ref_model is not None:
                with torch.no_grad():
                    ref_logits = ref_model(
                        input_ids=complete_prompts.to(ref_model.device)[:,:prompt_length],
                        decoder_input_ids=complete_prompts.to(ref_model.device)[:,prompt_length:]
                    ).logits
                    ref_logits /= temperature
            else:
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_logits = model(
                            input_ids=complete_prompts.to(model.device)[:,:prompt_length],
                            decoder_input_ids=complete_prompts.to(model.device)[:,prompt_length:]
                        ).logits
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

            return loss

    # Update the scheduler every rl step no matter the epochs
    if scheduler:
        scheduler.step()

# Cosine scheduler with warmup
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

# HYPERPARAMETERS

max_steps = 256*8 # FIXME fast test
sims_per_prompt = 4 # FIXME low mem
rl_steps = max_steps // sims_per_prompt
minibatch_size = 8
update_epochs = 1
policy_lr = 5e-6
kl_penalty_coef = 0.04
warmup_steps = 25

save_adapter_path = 'models/grpo_policy_model_nllb'
base_model_name = "wayuu-spanish/models/nllb_wayuu_esp_sin_dict_1_3B-V2"
best_adapter_path = 'models/best_policy_model_nllb'
src_lang = "spa_Latn"
tgt_lang = "way_Latn"
spanish_train_file = 'datasets/train.es.txt'
wayuu_train_file = 'datasets/train.guc.txt'
spanish_val_file = 'datasets/dev.es.txt'
wayuu_val_file = 'datasets/dev.guc.txt'
update_ref_model_steps = None
gae_lambda = 1.0
normalize_advantage=False
temperature=0.9 # Temperature for the generations
lower_clip=0.8
upper_clip=1.2
dr_grpo = True
no_kl=True
max_new_tokens=256 # FIXME poquita memoria GPU
accum_grad_steps = 4

logger.info(f'Hyperparameters:\nupdate_epochs:{update_epochs}\nrl_steps:{rl_steps}\nsims_per_prompt:{sims_per_prompt}\nminibatch_size:{minibatch_size}\npolicy_lr:{policy_lr}\nwarmup_steps:{warmup_steps}\ngae_lambda: {gae_lambda}\nnormalize advantage:{normalize_advantage}\nlower_clip:{lower_clip}\nupper_clip:{upper_clip}\nkl_penalty_coef:{kl_penalty_coef}\ntemperature:{temperature}\ndr_grpo:{dr_grpo}\nno_kl={no_kl}\n')

config = LoraConfig(
    r=32, # Rank de las matrices A y B
    lora_alpha=64, # Factor de regularización de las matrices A y B
    target_modules= [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # lora_dropout=0.05, # Dropout de las matrices A y B
    # bias="none", # No se añade bias a las capas lineales
    task_type="SEQ_2_SEQ_LM" # Tipo de tarea
)

# EXECUTION

model, tokenizer = get_policy_model(base_model_name, src_lang, tgt_lang)
# ref_model, _ = get_policy_model()
# ref_model.eval()

# Use LoRA to finetune the policy model
model = get_peft_model(model, config) # FIXME COMMENTED TO FAST TEST

optimizer = torch.optim.AdamW(model.parameters(), lr=policy_lr, betas=(0.9, 0.99), weight_decay=0.1)
# scheduler = CosineAnnealingLR(optimizer, T_max=rl_steps)
scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps, rl_steps)

ref_model = None

# Load the dataset
dataset = TextDataset(spanish_train_file, wayuu_train_file)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load validation dataset
validation_dataset = TextDataset(spanish_train_file, wayuu_train_file)

# Training loop
try:
    model.eval()
    acc = eval_translations(model, tokenizer, validation_dataset, batches=10, batch_size=4, generate_fn=partial(generate_batch_completion))
    logger.info(f'Evaluation before training: {acc}')
    model.train()

    max_performance = acc

    old_model = None

    rl_step = 0
    accumulated_grad_steps = 0
    loss = None
    while rl_step < rl_steps:
        logger.info(f'rl_step: {rl_step+1:,}')
        spa_sample, wayuu_sample = next(iter(dataloader))
        spa_sample, wayuu_sample = spa_sample[0], wayuu_sample[0]
        generations, rewards, is_terminal, complete_prompts, prompt_length = translation_simulation(model, sims_per_prompt, temperature=temperature, spa=spa_sample, wayuu=wayuu_sample, max_new_tokens=max_new_tokens)
        advantanges = compute_advantages(rewards, is_terminal, gae_lambda=gae_lambda, dr_grpo=dr_grpo)
        if (advantanges == 0).all().item():
            torch.cuda.empty_cache() # FIXME se comia toda la GPU rip
            gc.collect()
            continue

        logger.info('Updating policy')
        logger.debug(f'Learning rate: {scheduler.get_lr()}')
        loss = update_policy(model, ref_model, old_model, optimizer, is_terminal, advantanges, complete_prompts, prompt_length, generations, minibatch_size, update_epochs, scheduler=scheduler, normalize_advantage=normalize_advantage, lower_clip=lower_clip, upper_clip=upper_clip, dr_grpo=dr_grpo, no_kl=no_kl, temperature=temperature)
        loss = loss / accum_grad_steps
        accumulated_grad_steps += 1
        loss.backward()
 
        if accumulated_grad_steps == accum_grad_steps:
            # Update the policy weights
            nn.utils.clip_grad_norm_(model.parameters(), 0.1) # Avoid large gradients
            optimizer.step()
            optimizer.zero_grad()
        else:
            accumulated_grad_steps = 0

        # Track progress on specific task
        if (rl_step+1)%10 == 0:
            model.eval()
            with torch.no_grad():
                acc = eval_translations(model, tokenizer, validation_dataset, batches=10, batch_size=4, generate_fn=partial(generate_batch_completion))
            logger.info(f'Evaluation on rl step {rl_step+1:,}: {acc}')
            model.train()

            # Save the model if the performance is better
            if acc > max_performance:
                max_performance = acc
                logger.info(f'Saving model with performance {max_performance}')
                model.save_pretrained(best_adapter_path)
                logger.info(f'Best model saved to {best_adapter_path}')

        if update_ref_model_steps is not None and (rl_step+1)%update_ref_model_steps == 0:
            ref_model = copy.deepcopy(model).eval() # Update the ref model
        
        torch.cuda.empty_cache() # FIXME se comia toda la GPU rip
        gc.collect()
        rl_step += 1
        print("RL_STEP", rl_step)

except KeyboardInterrupt:
    pass

model.eval()
with torch.no_grad():
    acc = eval_translations(model, tokenizer, validation_dataset, batches=10, batch_size=4, generate_fn=partial(generate_batch_completion))
logger.info(f'Evaluation after training: {acc}')
model.save_pretrained(save_adapter_path)

end_time = time.time()
execution_time = end_time - start_time
logger.info(f'Total execution time in minutes: {execution_time/60:.2f}')