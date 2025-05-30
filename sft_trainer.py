import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from torch.utils.data import Dataset, DataLoader
import torch
import tqdm
import os
import logging
import time
from logging import getLogger
import gc
from torch.amp import GradScaler
from peft import PeftModel
from huggingface_hub import login

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

login(token = "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")

start_time = time.time()

logger = getLogger(__name__)
logsdir = 'logs'
logfile_name = 'sft_qwen7b.log'
logpath = os.path.join(logsdir, logfile_name)
if os.path.exists(logpath):
  os.remove(logpath)
logging.basicConfig(filename=logpath, encoding='utf-8', level=logging.DEBUG)

# Read the dataset
dataset_path = 'datasets/sft_dataset2.pkl'
# Create the dataset
class TranslationDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'rb') as file:
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt, answer = self.data[idx]
        
        return prompt, answer
# Load the dataset
dataset = TranslationDataset(dataset_path)
# Create the dataloader
batch_size = 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Load the model
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:2", 
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

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

# Use LoRA to finetune the model
model.gradient_checkpointing_enable()
model = get_peft_model(model, config)

best_checkpoint_name = "models/sft_base_qwen7b_tools"
min_loss = float("inf")

def tokenize_batch(batch, tokenizer):
    batch_formatted = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for prompt in batch[0]
    ]

    # build inputs and targets
    inputs = tokenizer.apply_chat_template(
        batch_formatted,
        tokenize=False,
        add_generation_prompt=True,
    )
    answers = [ans + tokenizer.eos_token for ans in batch[1]]

    completed = [inp + ans for inp, ans in zip(inputs, answers)]
    comp_enc = tokenizer(completed, return_tensors="pt", padding=True, padding_side="left")

    return comp_enc

def train_batch(batch, model, tokenizer, optimizer, device, tokenized_batch):
    # everything lives in this local scope
    # batch_formatted = [
    #     [
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": prompt},
    #     ]
    #     for prompt in batch[0]
    # ]

    # # build inputs and targets
    # inputs = tokenizer.apply_chat_template(
    #     batch_formatted,
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )
    answers = [ans + tokenizer.eos_token for ans in batch[1]]

    answers_encoded = tokenizer(answers)
    # completed = [inp + ans for inp, ans in zip(inputs, answers)]
    # comp_enc = tokenizer(completed, return_tensors="pt", padding=True, padding_side="left")
    comp_enc = tokenized_batch

    x   = comp_enc["input_ids"][:, :-1].to(device)
    xm  = comp_enc["attention_mask"][:, :-1].to(device)
    y   = comp_enc["input_ids"][:,  1:].to(device)
    ym  = comp_enc["attention_mask"][:,  1:].to(device)

    # mask out non-answer tokens
    mask = torch.zeros_like(ym)
    for i, enc in enumerate(answers_encoded.input_ids):
        mask[i, -len(enc):] = 1
    y_masked = y.masked_fill(mask == 0, -100)

    # forward / backward
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(x, attention_mask=xm).logits
        loss   = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y_masked.view(-1),
        )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # loss.backward()
    # optimizer.step()
    return loss.detach().cpu().item()

def flush_gpu(model=None, optimizer=None, locals_dict=None):
    # 1) delete any local tensors you know
    if locals_dict:
        for name in list(locals_dict):
            obj = locals_dict[name]
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                del locals_dict[name]
    # 2) clear optimizer state (if provided)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        # remove any internal optimizer tensors
        del optimizer
    # 3) move model to CPU (if you just want to offload it)
    if model is not None:
        model.cpu()
        del model
    # 4) collect everything else
    gc.collect()
    torch.cuda.empty_cache()
    # on newer PyTorch versions you can also:
    if hasattr(torch.cuda, 'ipc_collect'):
        torch.cuda.ipc_collect()

# Train the model
learning_rate = 1e-4
model.save_pretrained(best_checkpoint_name)
model.train()
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
num_epochs = 1
acumm_loss = []
# model = torch.compile(model)
oom_error = False
for epoch in range(num_epochs):
    step = 0
    for batch in tqdm.tqdm(dataloader):
        try:
            tokenized_batch = tokenize_batch(batch, tokenizer)
            if tokenized_batch['input_ids'].shape[1] > 768:
                logger.warning(f"Skipping batch for being too big. Size: {tokenized_batch['input_ids'].shape}")
                continue
            loss = train_batch(batch, model, tokenizer, optimizer, model.device, tokenized_batch)
            
            acumm_loss.append(loss)

            if (step+1) % 20 == 0 or step == 0:
                avg_loss = sum(acumm_loss) / len(acumm_loss)
                logger.info(f"Epoch {epoch}, Step {step+1}, Loss: {avg_loss:.4f}")
                if len(acumm_loss) > 15:
                    if avg_loss < min_loss:
                        min_loss = avg_loss
                        model.save_pretrained(best_checkpoint_name)
                        logger.info(f"Best checkpoint saved to {best_checkpoint_name}")
                acumm_loss = []
            step += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"CUDA OOM - skipping batch")
                tokenized_batch = tokenize_batch(batch, tokenizer)['input_ids']
                logger.warning(f'Batch size when OOM error: {tokenized_batch.shape}')
                # for p in model.parameters():
                #     if p.grad is not None:
                #         del p.grad  # free some memory
                # del batch
                # flush_gpu(model=model, optimizer=optimizer, locals_dict=locals())
                # gc.collect(1)
                # torch.cuda.empty_cache()
                # torch.cuda.ipc_collect()
                # time.sleep(1)
                # logger.debug(torch.cuda.memory_stats())
                # logger.debug(torch.cuda.memory_summary())
                # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
                # tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Load the PEFT model
                # model = PeftModel.from_pretrained(model, best_checkpoint_name)
                # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                # continue
                oom_error = True
                break
            else:
                raise e

        # Release GPU memory
        # del x, x_attention_mask, y, y_attention_mask, loss_mask, logits, loss
        torch.cuda.empty_cache()
        if oom_error:
            break
    logger.info(f"Epoch {epoch} completed.")

# Save the model
model.save_pretrained("models/sft_base_qwen7b_tools")
logger.info(f"Model saved to models/sft_base_llama_tools")

end_time = time.time()
execution_time = end_time - start_time
logger.info(f'Total execution time in minutes: {execution_time/60:.2f}')

# from torch.cuda.amp import autocast, GradScaler

# # before training:
# scaler = GradScaler()
# model.gradient_checkpointing_enable()

# # inside train_batch, wrap forward+loss inside `autocast()`, and use `scaler.scale(loss).backward()`
# with autocast():
#     logits = model(x, xm).logits
#     loss   = F.cross_entropy(...)

# scaler.scale(loss).backward()
# scaler.step(optimizer)
# scaler.update()