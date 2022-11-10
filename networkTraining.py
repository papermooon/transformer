from torch.autograd import Variable
from torch.utils.data import DataLoader
from hyperPara import *
from process import CommentaryDataset, en_tokens, zh_tokens, commentary_collate_fn
import math
import torch.nn as nn


# PE采用公式计算而非训练得到
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=max_length):
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model)  # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1)  # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        # torch.arange(start=1.0,end=6.0)和range区别,arrange生成浮点数值为start~end的数组但不包括end
        # unsqueeze(i)压缩，增加维数，在第i+1个维度上再加一维
        div_term = torch.exp(torch.arange(0, d_model, 2) *  # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)  # size = [batch, L, d_model]
        return self.dropout(x)  # size = [batch, L, d_model]


class TranslateModel(nn.Module):
    def __init__(self, d_model, src_vocab, tgt_vocab, drop_rate=0.1):
        super(TranslateModel, self).__init__()
        # embedding and PE
        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        self.PE = PositionalEncoding(d_model, drop_rate)
        # transformer
        self.transformer = nn.Transformer(d_model, dropout=drop_rate, batch_first=True)

        self.linear = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        # src: 原batch后的句子，例如[[0, 12, 34, .., 1, 2, 2, ...], ...]
        # tgt: 目标batch后的句子，例如[[0, 74, 56, .., 1, 2, 2, ...], ...]

        return


dataset = CommentaryDataset(en_tokens, zh_tokens)
train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=commentary_collate_fn)
