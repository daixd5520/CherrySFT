import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import torch
import argparse
import os
import json
import pickle

# 解析命令行参数
parser = argparse.ArgumentParser(description="Process model responses.")
parser.add_argument('--file', type=int, required=True, help="File number (1, 2, or 3)")
args = parser.parse_args()

# 根据命令行参数选择文件
file_number = args.file
file_path = f'/home/dxd/SFT/All-Utils/infer_res/llama2-7b-lora/wizard/1006/checkpoint/{file_number}/checkpoint-removedup.pkl'
checkpoint_dir = f'/home/dxd/SFT/All-Utils/score_res/checkpoints/{file_number}'
checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.json')
output_path = f'/home/dxd/SFT/All-Utils/score_res/scoring_res{file_number}.pkl'

# 确保检查点目录存在
os.makedirs(checkpoint_dir, exist_ok=True)
device = torch.device("cuda:0")
# 加载模型和tokenizer
model_name = "/home/dxd/SFT/URM-LLaMa-3.1-8B-half"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    device_map='auto',
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device)   # 确保模型在GPU上运行
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open(file_path, 'rb') as f:
    # 加载pkl文件
    df = pickle.load(f)
# 将列表转换为DataFrame
df = pd.DataFrame(df)  
    

# 检查是否存在检查点文件
start_index = 0
results = []
if os.path.exists(checkpoint_path):
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
        start_index = checkpoint_data['index']
        results = checkpoint_data['results']

# 批量大小
batch_size = 2

# 使用tqdm添加进度条
for batch_start in tqdm(range(start_index, df.shape[0], batch_size), desc="Processing Batches"):
    batch_end = min(batch_start + batch_size, df.shape[0])
    batch_df = df.iloc[batch_start:batch_end]
    
    prompts = batch_df['Instruction'].tolist()
    
    # 初始化得分列表
    scores1_list = []
    scores2_list = []
    
    # 对每个epo的五条结果进行评分
    for epo in ['1epo', '2epo']:
        for res_num in range(1, 6):
            responses = batch_df[f'{epo}-res{res_num}'].tolist()
            resp_list = [[{"role": "user", "content": prompt}, {"role": "assistant", "content": response}] for prompt, response in zip(prompts, responses)]
            
            # 格式化和tokenize对话
            resp_list = [tokenizer.apply_chat_template(resp, tokenize=False) for resp in resp_list]
            resp_batch = tokenizer(resp_list, return_tensors="pt", padding=True, truncation=True).to('cuda')
            
            # 批处理推理
            with torch.no_grad():
                scores = model(resp_batch['input_ids'], attention_mask=resp_batch['attention_mask']).logits[:, 0].tolist()
            
            if epo == '1epo':
                scores1_list.append(scores)
            else:
                scores2_list.append(scores)
    
    # 计算每个epo的平均得分
    scores1_avg = [sum(scores) / len(scores) for scores in zip(*scores1_list)]
    scores2_avg = [sum(scores) / len(scores) for scores in zip(*scores2_list)]
    
    # 计算分数差
    score_diffs = [score2 - score1 for score1, score2 in zip(scores1_avg, scores2_avg)]
    
    # 将结果添加到列表中
    for i, row in batch_df.iterrows():
        results.append({
            'Index': row['Index'],
            'Instruction': row['Instruction'],
            '1epo-Bo5得分': scores1_avg[i - batch_start],
            '2epo-Bo5得分': scores2_avg[i - batch_start],
            '分差': score_diffs[i - batch_start],
            '1epo-res1得分': scores1_list[0][i - batch_start],
            '1epo-res2得分': scores1_list[1][i - batch_start],
            '1epo-res3得分': scores1_list[2][i - batch_start],
            '1epo-res4得分': scores1_list[3][i - batch_start],
            '1epo-res5得分': scores1_list[4][i - batch_start],
            '2epo-res1得分': scores2_list[0][i - batch_start],
            '2epo-res2得分': scores2_list[1][i - batch_start],
            '2epo-res3得分': scores2_list[2][i - batch_start],
            '2epo-res4得分': scores2_list[3][i - batch_start],
            '2epo-res5得分': scores2_list[4][i - batch_start],
        })
    
    # 保存当前进度和结果到检查点文件
    checkpoint_data = {
        'index': batch_end,
        'results': results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint_data, f)

# 将结果保存到pickle文件
with open(output_path, 'wb') as f:
    pickle.dump(results, f)

# 删除检查点文件
# os.remove(checkpoint_path)

print(f"Scoring completed and results saved to {output_path}")