import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
MY_TOKEN = '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' # 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

config.k_bits = 2 # current support 2/4 bit for KV Cache
config.v_bits = 2 # current support 2/4 bit for KV Cache
config.group_size = 32
config.residual_length = 32 # the number of recent fp16 tokens
config.use_flash = True
# CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

model = LlamaForCausalLM_KIVI.from_pretrained(
    pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    config=config,
    # cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    token=MY_TOKEN
)

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf', 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

# Inference
# e.g., model.generate(...)