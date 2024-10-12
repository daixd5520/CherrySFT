"""
对于merge完的1epo模型和2epo模型进行在训练数据集上的推理
具有断点续推功能
python generate_bo5.py --part 1/2/3 --model1 {model1-path} --model2 {model2-path}
"""
import json
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import argparse
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--part",
        type=str,
        default="1",
        help="The part num of Alpaca(1/2/3).",
    )
    parser.add_argument(
        "--model1",
        type=str,
        default="/home/dxd/SFT/wizard-resmodels/LoRA-merged/1epo",
        help="model1 path.",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="/home/dxd/SFT/wizard-resmodels/LoRA-merged/2epo",
        help="model2 path.",
    )
    
    args = parser.parse_args()
    return args

args = parse_args()
part = int(args.part) 
output_file = f'./infer_res/llama2-7b-lora-wizard/wizard-part{part}_bo5.xlsx'
checkpoint_dir = f'./infer_res/llama2-7b-lora/wizard/1006/checkpoint/{part}'

if torch.cuda.device_count() < 2:
    raise RuntimeError("需要至少2个GPU来运行此代码。")

paths = [args.model1, args.model2]

tokenizer = AutoTokenizer.from_pretrained(paths[0])
tokenizer.pad_token = tokenizer.eos_token

# 设置第一个模型使用 GPU 0，第二个模型使用 GPU 1
device0 = torch.device("cuda:0")
device1 = torch.device("cuda:1")

# Load the models onto different GPUs
model_1epo = AutoModelForCausalLM.from_pretrained(paths[0]).bfloat16().to(device0)
model_2epo = AutoModelForCausalLM.from_pretrained(paths[1]).bfloat16().to(device1)

def gen_batch(contents, model, device, num_responses=5):
    input_texts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{content}\n\n### Response:\n" for content in contents]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)  # 将输入张量移到对应的 GPU
    input_lengths = [inputs['input_ids'][i].shape[0] for i in range(len(contents))]
    
    generated_answers = [[] for _ in range(len(contents))]
    for _ in range(num_responses):
        outputs = model.generate(**inputs, max_new_tokens=100)
        for i in range(len(contents)):
            generated_answer = tokenizer.decode(outputs[i][input_lengths[i]:], skip_special_tokens=True)
            generated_answers[i].append(generated_answer)
    
    return generated_answers

# Load Alpaca data
with open('/home/dxd/SFT/LLaMA-Factory/dld_data/alpaca_evol_instruct_70k.json', 'r') as file:
    alpaca_data = json.load(file)

# 计算数据的1/3长度
third_length = len(alpaca_data) // 3

# 截取part数据
alpaca_data_third = alpaca_data[(part-1)*third_length: part*third_length]

# Prepare the checkpoint directory
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pkl')

# Check if checkpoint file exists
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        results = pickle.load(f)
else:
    results = []

# Set batch size
batch_size = 20

# Process data with progress bar and save after each inference
for idx in tqdm(range(0, len(alpaca_data_third), batch_size), desc=f"Processing {part}th Third of Alpaca Data"):
    batch_instructions = [entry['instruction'] for entry in alpaca_data_third[idx:idx + batch_size]]
    
    # Generate responses on different GPUs
    response_1epo_batch = gen_batch(batch_instructions, model_1epo, device0)
    response_2epo_batch = gen_batch(batch_instructions, model_2epo, device1)
    
    # Append the result to the results list
    for i, instruction in enumerate(batch_instructions):
        results.append({
            'Index': idx + i + (part-1)*third_length,
            'Instruction': instruction,
            '1epo-res1': response_1epo_batch[i][0],
            '1epo-res2': response_1epo_batch[i][1],
            '1epo-res3': response_1epo_batch[i][2],
            '1epo-res4': response_1epo_batch[i][3],
            '1epo-res5': response_1epo_batch[i][4],
            '2epo-res1': response_2epo_batch[i][0],
            '2epo-res2': response_2epo_batch[i][1],
            '2epo-res3': response_2epo_batch[i][2],
            '2epo-res4': response_2epo_batch[i][3],
            '2epo-res5': response_2epo_batch[i][4],
        })
    
    # Save the results to the pickle file
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(results, f)

# Convert the results to a DataFrame and save to Excel
result_df = pd.DataFrame(results)
result_df.to_excel(output_file, index=False)

print(f"Results are saved to {output_file}.")