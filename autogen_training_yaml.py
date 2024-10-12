"""

自动生成LLaMA-Factory的LoRA训练YAML文件，/home/dxd/SFT/LLaMA-Factory/examples/train_lora/Wizard/llama2_7b_lora_sft_Wizard_remove4k.yaml


python autogen_training_yaml.py --yaml_prefix '/home/dxd/SFT/LLaMA-Factory/examples/train_lora/Wizard/llama2_7b_lora_sft_Wizard_remove'  \
    --model_path '/home/dxd/SFT/models/Llama-2-7b-hf' \
    --data_prefix 'wizard-70k' \
    --remove_count '[4,8,12,16,20,24,28,32]' \
    --save_path '/home/dxd/SFT/All-Utils/lora_res_after_choose_data/remove'



"""
import os
import yaml
import sys
import argparse

def generate_yaml(yaml_prefix, model_path, data_prefix, remove_count, save_path):
    for count in remove_count:
        # 构建 YAML 内容的各个部分
        yaml_sections = {
            "model": {
                "model_name_or_path": model_path
            },
            "method": {
                "stage": "sft",
                "do_train": True,
                "finetuning_type": "lora",
                "lora_target": "all"
            },
            "dataset": {
                "dataset": f"{data_prefix}{count}",
                "template": "llama2",
                "cutoff_len": 1024,
                "max_samples": 1000,
                "overwrite_cache": True,
                "preprocessing_num_workers": 16
            },
            "output": {
                "output_dir": f"{save_path}{count}k",
                "logging_steps": 10,
                "save_strategy": "epoch",
                "save_steps": 500
            },
            "train": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 8,
                "learning_rate": 1.0e-4,
                "num_train_epochs": 3.0,
                "lr_scheduler_type": "cosine",
                "warmup_ratio": 0.1,
                "bf16": True,
                "ddp_timeout": 180000000
            },
            "eval": {
                "val_size": 0.1,
                "per_device_eval_batch_size": 1,
                "eval_strategy": "steps",
                "eval_steps": 500
            }
        }

        # 定义每个部分前的注释
        comments = {
            "model": "### model",
            "method": "### method",
            "dataset": "### dataset",
            "output": "### output",
            "train": "### train",
            "eval": "### eval"
        }

        # 初始化 YAML 字符串
        yaml_str = ""

        # 按顺序添加注释和对应的内容
        for section in ["model", "method", "dataset", "output", "train", "eval"]:
            yaml_str += f"{comments[section]}:\n"
            # 使用缩进来表示层级结构
            section_yaml = yaml.dump(yaml_sections[section], default_flow_style=False, sort_keys=False)
            # 增加缩进
            indented_section = '\n'.join(['  ' + line if line.strip() else line for line in section_yaml.split('\n')])
            yaml_str += f"{indented_section}\n"

        # 定义文件名
        file_name = f"{yaml_prefix}{count}k.yaml"
        with open(file_name, 'w') as file:
            file.write(yaml_str)

        print(f"Generated {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YAML files based on command line arguments.")
    parser.add_argument("--yaml_prefix", type=str, help="Path prefix of your yaml.")
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--data_prefix", type=str, help="Prefix for the dataset.")
    parser.add_argument("--remove_count", type=str, help="List of remove counts, e.g., '[4,5,6,7,8]'.")
    parser.add_argument("--save_path", type=str, help="Prefix for the save path.")

    args = parser.parse_args()

    remove_count = list(map(int, args.remove_count.strip('[]').split(',')))

    generate_yaml(args.yaml_prefix, args.model_path, args.data_prefix, remove_count, args.save_path)
