import os
import json
import tqdm
import tarfile
import shutil
from modelscope import dataset_snapshot_download
from datasets import load_dataset

# 检查是否存在预训练数据文件
input_file = 'mobvoi_seq_monkey_general_open_corpus.jsonl'
if not os.path.exists(input_file):
    print(f"正在下载预训练数据文件: {input_file}")
    try:
        # 从modelscope下载数据集
        print("从modelscope下载数据集...")
        dataset_path = dataset_snapshot_download('ddzhu123/seq-monkey', local_dir='train_data')
        # 查找并解压tar.bz2文件
        tar_file = None
        for root, dirs, files in os.walk('train_data'):
            for file in files:
                if 'mobvoi_seq_monkey_general_open_corpus' in file and file.endswith('.tar.bz2'):
                    tar_file = os.path.join(root, file)
                    break
            if tar_file:
                break

        if tar_file:
            print(f"解压文件: {tar_file}")
            with tarfile.open(tar_file, 'r:bz2') as tar:
                tar.extractall()
            print(f"数据文件已解压")
        else:
            # 如果没有找到通用语料库，查找其他可用的jsonl文件
            print("未找到通用语料库，查找其他可用数据...")
            for root, dirs, files in os.walk('train_data'):
                for file in files:
                    if file.endswith('.jsonl'):
                        src = os.path.join(root, file)
                        shutil.copy(src, input_file)
                        print(f"使用数据文件: {file}")
                        break
                    elif file.endswith('.tar.bz2'):
                        print(f"解压: {file}")
                        with tarfile.open(os.path.join(root, file), 'r:bz2') as tar:
                            tar.extractall()
                        # 再查找解压后的jsonl文件
                        for extracted_file in os.listdir('.'):
                            if extracted_file.endswith('.jsonl'):
                                shutil.copy(extracted_file, input_file)
                                print(f"使用解压后的数据文件: {extracted_file}")
                                break
                        if os.path.exists(input_file):
                            break
                if os.path.exists(input_file):
                    break
    except Exception as e:
        print(f"下载失败: {e}")
        raise

# 检查SFT数据
sft_data_file = 'BelleGroup_sft.jsonl'
if not os.path.exists(sft_data_file):
    print(f"正在下载SFT数据文件: {sft_data_file}")
    try:
        # 从HuggingFace下载数据集
        print("从HuggingFace下载BelleGroup数据集...")
        dataset = load_dataset('BelleGroup/train_3.5M_CN', cache_dir='bellegroup_cache')

        # 转换格式并保存
        def convert_message(conversations):
            message = [
                {"role": "system", "content": "你是一个AI助手"},
            ]
            for item in conversations:
                if item['from'] == 'human':
                    message.append({'role': 'user', 'content': item['value']})
                elif item['from'] == 'assistant':
                    message.append({'role': 'assistant', 'content': item['value']})
            return message

        print(f"保存数据到 {sft_data_file}")
        with open(sft_data_file, 'w', encoding='utf-8') as f:
            for item in tqdm.tqdm(dataset['train'], desc="转换SFT数据"):
                message = convert_message(item['conversations'])
                f.write(json.dumps(message, ensure_ascii=False) + '\n')

        print(f"SFT数据下载完成: {sft_data_file}")
    except Exception as e:
        print(f"下载失败: {e}")
        raise


# 1 处理预训练数据
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

input_file = 'mobvoi_seq_monkey_general_open_corpus.jsonl'

with open('seq_monkey_datawhale.jsonl', 'a', encoding='utf-8') as pretrain:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm.tqdm(data, desc="Processing lines", leave=False):
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

