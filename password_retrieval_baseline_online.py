import requests
import json

API_TOKEN = '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'
MODEL = "meta-llama/Llama-3.2-1B"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

def query_model(prompt):
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
        }

    data = {
        "inputs": prompt
    }
    response = requests.post(
        API_URL,
        headers=headers,
        json=data
    )
    # print(response.json())
    return response.json()

file_name = "passkey_examples.jsonl"
method_name = "Base LLaMA API"

print("=========="*2 + f"**{method_name}**" + "=========="*2)

for line in open(file_name, "r"):
    example = json.loads(line)
    prompt_postfix = "What is the pass key? The pass key is "
    prompt = example["input"] + prompt_postfix
    
    print("-----------------------------------")
    # print("Prompt:", prompt)
    print("Passkey target:", example["target"])

    # Query the model through the API
    result = query_model(prompt)
    answer = result[0]["generated_text"] if "generated_text" in result[0] else "Error in response"
    print(answer)
    print("-----------------------------------\n")
