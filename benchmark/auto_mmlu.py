"""

自动化执行mmlu.需在deepeval环境下

python auto_mmlu.py --cvd 2 3 4 --model_id 4 8 16

参数说明：
cvd:CUDA_VISIBLE_DEVICES
model_id: model后缀

"""
import subprocess
import multiprocessing
import argparse

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        print(line.decode('utf-8'), end='')
    process.wait()

def main(args):
    gpu_ids = args.cvd
    model_ids = args.model_id

    if len(gpu_ids) != len(model_ids):
        print("Error: The number of GPUs and models must be the same.")
        return

    commands = []
    for gpu_id, model_id in zip(gpu_ids, model_ids):
        command = f"""
        MODEL=/home/dxd/SFT/All-Utils/lora_res_after_choose_data/merged/{model_id}
        device={gpu_id}
        CUDA_VISIBLE_DEVICES=$device python /home/dxd/SFT/llmuses/run.py\
        --model $MODEL \
        --datasets mmlu \
        --outputs /home/dxd/SFT/All-Utils/benchmark/benchmark_res/mmlu/{model_id} \
        --limit 10
        """
        commands.append(command)

    processes = []
    for command in commands:
        p = multiprocessing.Process(target=run_command, args=(command,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run run.py on multiple GPUs with different models.")
    parser.add_argument('--cvd', nargs='+', type=int, required=True, help="List of CUDA_VISIBLE_DEVICES GPU IDs")
    parser.add_argument('--model_id', nargs='+', type=int, required=True, help="List of model IDs")

    args = parser.parse_args()
    main(args)