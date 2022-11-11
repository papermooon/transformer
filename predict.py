import math
from tkinter import Variable
import torch.nn as nn
from hyperPara import *
from process import en_vocab, en_tokenizer, zh_vocab


# PE采用公式计算而非训练得到
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=max_length):
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model).to(device)  # 建立空表，每行代表一个词的位置，每列代表一个编码位
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
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
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

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(device)

        src_key_mask = TranslateModel.get_key_padding_mask(src)
        tgt_key_mask = TranslateModel.get_key_padding_mask(tgt)

        src = self.src_embedding(src)
        src = self.PE(src)

        tgt = self.tgt_embedding(tgt)
        tgt = self.PE(tgt)

        # print(src.device, tgt.device, tgt_mask.device, src_key_mask.device, tgt_key_mask.device)
        output = self.transformer(
            src, tgt, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_mask,
            tgt_key_padding_mask=tgt_key_mask,
        )

        output = self.linear(output)

        return output

    @staticmethod
    def get_key_padding_mask(tokens):
        key_padding_mask = torch.zeros(tokens.size(), dtype=torch.bool).to(device)
        key_padding_mask[tokens == 2] = True
        return key_padding_mask


def translate(src: str):
    model = torch.load(model_dir / checkpoint)
    model.eval()
    src = torch.tensor([0] + en_vocab(en_tokenizer(src)) + [1]).unsqueeze(0).to(device)
    tgt = torch.tensor([[0]]).to(device)
    for i in range(max_length):
        # 进行transformer计算
        out = model(src, tgt)
        # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
        predict = out[:, -1]
        # 找出最大值的index
        y = torch.argmax(predict, dim=1)
        # 和之前的预测结果拼接到一起
        tgt = torch.concat([tgt, y.unsqueeze(0)], dim=1)
        if y == 1:
            break
    # 将预测tokens拼起来
    tgt = ''.join(zh_vocab.lookup_tokens(tgt.squeeze().tolist())).replace("<bos>", "").replace("<eos>", "")
    return tgt


print(translate(
    "For example, lobbying education leaders to keep girls in school longer has contributed to providing young people with the knowledge and agency to make smart decisions about when and with whom to negotiate safe sex."))
print(translate("you are a shit translation system!"))
print(translate("The NCD and AIDS communities can learn from one another."))
print(translate("And, in fact, albedo modification would undoubtedly make some things worse."))
print(translate("One reason for this is concern about the diversion of resources from other approaches."))
print(translate("In 1997,I am a pig."))
print(translate("maybe i made a huge mistake."))
print(translate("it is a failure."))
