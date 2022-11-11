from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from hyperPara import *
from process import CommentaryDataset, en_tokens, zh_tokens, commentary_collate_fn, en_vocab, zh_vocab
import math
import torch.nn as nn
from tqdm import tqdm


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


# 载入模型
if checkpoint is not None:
    model = torch.load(model_dir / checkpoint)
else:
    model = TranslateModel(128, en_vocab, zh_vocab)
model.to(device)

dataset = CommentaryDataset(en_tokens, zh_tokens)
train_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=commentary_collate_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criteria = nn.CrossEntropyLoss(ignore_index=2)
writer = SummaryWriter(log_dir='runs/CE_loss_plot')
current_step = 0

model.train()
for epoch in range(epochs):
    loop = tqdm(enumerate(train_iter), total=len(train_iter))
    for batch_data in train_iter:
        optimizer.zero_grad()

        raw, feed, label, valid_token = batch_data
        raw = raw.to(device)
        feed = feed.to(device)
        label = label.to(device)

        output = model(raw, feed)
        loss = criteria(output.contiguous().view(-1, output.size(-1)), label.contiguous().view(-1)) / valid_token
        loss.backward()
        optimizer.step()

        writer.add_scalar(tag="loss",  # 可以理解为图像的名字
                          scalar_value=loss.item(),  # 纵坐标的值
                          global_step=current_step  # 当前是第几次迭代，可以理解为横坐标的值
                          )
        current_step = current_step + 1

        loop.set_description("Epoch {}/{}".format(epoch+1, epochs))
        loop.set_postfix(loss=loss.item())
        loop.update(1)
    torch.save(model, model_dir / f"model_epoch_{epoch}.pt")
# src, tgt, tgt_y, n_tokens = next(iter(train_iter))
# src, tgt, tgt_y = src.to(device), tgt.to(device), tgt_y.to(device)
# print("src.size:", src.size())
# print("tgt.size:", tgt.size())
# print("tgt_y.size:", tgt_y.size())
# print("n_tokens:", n_tokens)
# print("src example:", src[0])
# print("tgt example:", tgt[0])
# tmp=model(src, tgt)
# y=torch.argmax(tmp,dim=2)
# tmp2=tmp.size()
# print(tmp,tmp2)
# print(y,y.size())
