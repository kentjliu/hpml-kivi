import warnings
warnings.filterwarnings("ignore")
import torch
import json
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from datasets import load_dataset

# Load the configuration for the base LLaMA model
config = LlamaConfig.from_pretrained("lmsys/vicuna-7b-v1.5-16k")
config.use_flash = True  # Use flash-attention if supported for long context inference
CACHE_DIR = "/scratch/cached_model"

# Initialize the base LLaMA model
model = LlamaForCausalLM.from_pretrained(
    pretrained_model_name_or_path="lmsys/vicuna-7b-v1.5-16k",
    config=config,
    # cache_dir=CACHE_DIR,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).cuda()

# Load the tokenizer
enc = AutoTokenizer.from_pretrained(
    'lmsys/vicuna-7b-v1.5-16k', 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama'
)

# Prepare the model for evaluation
model.eval()

# Define the file with examples and the model name for tracking
file_name = "passkey_examples.jsonl"
method_name = "Base LLaMA"

print("=========="*2 + f"**{method_name}**" + "=========="*2)

# Iterate through the examples in the JSONL file
for line in open(file_name, "r"):
    example = json.loads(line)
    prompt_postfix = "What is the pass key? The pass key is "
    prompt = example["input"] + prompt_postfix
    input_ids = enc(prompt, return_tensors="pt").input_ids.cuda()
    
    print("-----------------------------------")
    print(f"#Tokens of Prompt:", input_ids.shape[1], end=" ")
    print("Passkey target:", example["target"])

    # Generate output from the model
    tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
    answer = prompt_postfix + enc.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
    answer = answer.replace("\n", "\\n")
    answer = f"{method_name}:\n     [ {answer} ]"
    print(answer)
    print("-----------------------------------\n")
