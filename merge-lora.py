from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import torch

# 解析命令行参数
parser = argparse.ArgumentParser(description="Merge LoRA weights into a base model and save the merged model.")
parser.add_argument("--lora", type=str, required=True, help="Path to the LoRA model directory.")
parser.add_argument("--model", type=str, required=True, help="Path to the base model directory.")
parser.add_argument("--save", type=str, required=True, help="Path to save the merged model.")
args = parser.parse_args()

# 从命令行参数获取路径
LORA_PATH = args.lora
MODEL_PATH = args.model
SAVE_PATH = args.save

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
base_model.resize_token_embeddings(len(tokenizer))

# 加载 LoRA 模型
model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 合并 LoRA 权重到基础模型
model = model.merge_and_unload()

# 修正生成配置
if hasattr(model, "generation_config"):
    # model.generation_config.do_sample = True  # 设置为 True 以启用 temperature 和 top_p
    model.generation_config.temperature = None  # 或者移除 temperature
    model.generation_config.top_p = None  # 或者移除 top_p

# 将模型参数转换为半精度
model = model.half()

# 保存合并后的模型
model.save_pretrained(SAVE_PATH)

# 保存 tokenizer 和其配置
tokenizer.save_pretrained(SAVE_PATH)