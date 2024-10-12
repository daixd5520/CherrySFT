import pandas as pd
import json
import pickle

# 定义文件路径
RES_PATHS = [
    '/home/dxd/SFT/All-Utils/score_res/scoring_res1.pkl',
    '/home/dxd/SFT/All-Utils/score_res/scoring_res2.pkl',
    '/home/dxd/SFT/All-Utils/score_res/scoring_res3.pkl'
]

print("读取RM评分数据和合成中...")
dfs = []
for path in RES_PATHS:
    with open(path, 'rb') as f:
        data = pickle.load(f)
        # print(f"加载的数据来自 {path}: {data}")  # 更详细的打印信息
        df_temp = pd.DataFrame(data)  # 将每个数据转换为 DataFrame
        dfs.append(df_temp)           # 添加到 dfs 列表中

# 确保 dfs 不为空再进行合并
if dfs:
    df = pd.concat(dfs, ignore_index=True)
    print("读取合并完成")
    print(df.head())  # 打印前几行以确认
else:
    print("没有数据被加载和合并。")


# 按照 '分差' 列进行排序
df = df.sort_values(by='分差')
selected_cols = ['Index', '分差']
extracted_data = df[selected_cols]
print(f"#################################\n排序完成，df:\n{extracted_data}\n###########################################")

# 删去 '分差' 列中为负值的行
df = df[df['分差'] >= 0]

delete_counts = [4000, 8000, 12000, 16000, 20000, 24000, 28000, 32000]

file_prefix = '/home/dxd/SFT/All-Utils/wizard_chosen/del'

for count in delete_counts:
    to_delete_indices = df.head(count)['Index'].tolist()
    
    # 读取原始数据
    with open('/home/dxd/SFT/LLaMA-Factory/dld_data/alpaca_evol_instruct_70k.json', 'r') as f:
        data = json.load(f)
    
    # 逆序删除
    for index in sorted(to_delete_indices, reverse=True):
        del data[index]
    
    file_name = f'{file_prefix}{count}.json'
    with open(file_name, 'w') as f:
        # 保存
        json.dump(data, f, indent=4)
        print(f"删除{count}条数据的wizard数据集已保存至{file_name}...")