import os
import math

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import pad, log_softmax
from pathlib import Path
from tqdm import tqdm
from hyperPara import *
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# 划分数据集、测试集
def divide_sets(rebuild=False):
    total = get_row_number(en_filepath)
    training_set_size = int(total * 0.9)
    print("训练集:", training_set_size, "测试集:", total - training_set_size)
    if not rebuild:
        return
    else:
        print("正在划分数据集")
        en_tgt1 = open(training_set_en, "w", encoding="utf-8")
        en_tgt2 = open(testing_set_en, "w", encoding="utf-8")
        en_src = open(en_filepath, encoding="utf-8")

        zh_tgt1 = open(training_set_zh, "w", encoding="utf-8")
        zh_tgt2 = open(testing_set_zh, "w", encoding="utf-8")
        zh_src = open(zh_filepath, encoding="utf-8")

        ct = 0
        for line in en_src:
            ct = ct + 1
            if ct <= training_set_size:
                en_tgt1.write(line)
            else:
                en_tgt2.write(line)
        en_tgt1.close()
        en_tgt2.close()
        en_src.close()

        ct = 0
        for line in zh_src:
            ct = ct + 1
            if ct <= training_set_size:
                zh_tgt1.write(line)
            else:
                zh_tgt2.write(line)
        zh_tgt1.close()
        zh_tgt2.close()
        zh_src.close()


# 统计句子数量
def get_row_number(filename):
    ct = 0
    f = open(filename, encoding='utf-8')
    line = f.readline()
    while line:
        ct = ct + 1
        line = f.readline()
    f.close()
    return ct


# 中英文分词器
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")


def en_tokenizer(sentence):
    return tokenizer.encode(sentence, add_special_tokens=False).tokens


def zh_tokenizer(sentence):
    return list(sentence.replace(" ", "").strip())


# 中英文构造器
def yield_tokens(filename, mode):
    f = open(filename, encoding='utf-8')
    line = f.readline()
    while line:
        if mode == "zh":
            yield zh_tokenizer(line)
        if mode == "en":
            yield en_tokenizer(line)
        line = f.readline()
    f.close()


# 构造词典
def get_en_vocab(rebuild=False):
    if (not rebuild) and os.path.exists(en_vocab_path):
        en_vocab = torch.load(en_vocab_path, map_location="cpu")
    else:
        print("构建英文词典")
        en_vocab = build_vocab_from_iterator(
            yield_tokens(training_set_en, "en"),
            min_freq=1,
            specials=["<bos>", "<eos>", "<pad>", "<unk>"],
        )
        # 文本转index未知项用unk替换
        en_vocab.set_default_index(3)
        torch.save(en_vocab, en_vocab_path)
    return en_vocab


def get_zh_vocab(rebuild=False):
    if (not rebuild) and os.path.exists(zh_vocab_path):
        zh_vocab = torch.load(zh_vocab_path, map_location="cpu")
    else:
        print("构建中文词典")
        zh_vocab = build_vocab_from_iterator(
            yield_tokens(training_set_zh, "zh"),
            min_freq=1,
            specials=["<bos>", "<eos>", "<pad>", "<unk>"],
        )
        # 文本转index未知项用unk替换
        zh_vocab.set_default_index(3)
        torch.save(zh_vocab, zh_vocab_path)
    return zh_vocab


class CommentaryDataset(Dataset):
    def __init__(self, en_tokens, zh_tokens):
        self.en_tokens = en_tokens
        self.zh_tokens = zh_tokens

    def __getitem__(self, index):
        return self.en_tokens[index], self.zh_tokens[index]

    def __len__(self):
        return len(zh_tokens)


def get_token_list(token_path, src=None, mode="en", rebuild=False):
    if (not rebuild) and os.path.exists(token_path):
        return torch.load(token_path, map_location="cpu")
    else:
        token_list = []
        src_file = open(src, encoding="utf-8")
        if mode == "zh":
            print("构造中文token")
            for line in src_file:
                token_list.append(zh_vocab(zh_tokenizer(line)))
        else:
            print("构造英文token")
            for line in src_file:
                token_list.append(en_vocab(en_tokenizer(line)))
        torch.save(token_list, token_path)
    return token_list


def commentary_collate_fn(batch):
    bos = torch.tensor([0])
    eos = torch.tensor([1])
    raw_inputs = []
    feed_inputs = []

    # 添加bos eos
    for (raw_item, feed_item) in batch:
        raw_expand = torch.cat([bos, torch.tensor(raw_item, dtype=torch.int64), eos], dim=0)
        feed_expand = torch.cat([bos, torch.tensor(feed_item, dtype=torch.int64), eos], dim=0)

        # padding到最大句子长度,<pad>的索引为2
        raw_pad = pad(raw_expand, (0, max_length - len(raw_expand)), value=2)
        feed_pad = pad(feed_expand, (0, max_length - len(feed_expand)), value=2)

        raw_inputs.append(raw_pad)
        feed_inputs.append(feed_pad)

    # 原始输入，<bos><eos><pad>
    raw_inputs = torch.stack(raw_inputs)

    feed_inputs = torch.stack(feed_inputs)
    # 预测输出，去掉<bos>
    predict_outputs = feed_inputs[:, 1:]
    # 预测输入,去掉最后的token
    feed_inputs = feed_inputs[:, :-1]

    # 整个batch需要预测的token数量（pad不算）
    predict_token_sum = (predict_outputs != 2).sum()

    return raw_inputs, feed_inputs, predict_outputs, predict_token_sum


print("使用设备:", device)
divide_sets()
en_vocab = get_en_vocab()
zh_vocab = get_zh_vocab()
print("英文词典大小:", len(en_vocab))
print("中文词典大小:", len(zh_vocab))
# print(dict((i, zh_vocab.lookup_token(i)) for i in range(20)))
# print(dict((i, en_vocab.lookup_token(i)) for i in range(20)))

# zh_tokens = get_token_list(zh_token_path, src=training_set_zh, mode="zh", rebuild=True)
# en_tokens = get_token_list(en_token_path, src=training_set_en, mode="en", rebuild=True)
zh_tokens = get_token_list(zh_token_path)
en_tokens = get_token_list(en_token_path)

# print(zh_vocab.lookup_tokens(zh_tokens[1]))
# print(en_vocab.lookup_tokens(en_tokens[1]))
