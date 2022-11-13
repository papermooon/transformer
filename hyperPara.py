import torch
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
work_dir = Path("./dataset")
model_dir = Path("./model")
checkpoint = "model_epoch_19_11-13_06-34.pt"

# 中英文句子的文件路径
en_filepath = './dataset/news-commentary-v13.zh-en.en'
zh_filepath = './dataset/news-commentary-v13.zh-en.zh'

# 中英文数据集的文件路径
training_set_en = './dataset/training_set.en'
training_set_zh = './dataset/training_set.zh'
testing_set_en = './dataset/testing_set.en'
testing_set_zh = './dataset/testing_set.zh'

# 中英文词典的文件路径
en_vocab_path = './dataset/en_vocab.pt'
zh_vocab_path = './dataset/zh_vocab.pt'

# 中英文token的文件路径
en_token_path = './dataset/en_token.pt'
zh_token_path = './dataset/zh_token.pt'

max_length = 128
batch_size = 64
epochs = 20
