import math
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


def translate_bs(src: str, beam_size: int):
    model = torch.load(model_dir / checkpoint_next)
    model.eval()
    with torch.no_grad():
        base = torch.tensor([0] + en_vocab(en_tokenizer(src)) + [1]).unsqueeze(0).to(device)
        if base.shape[1] >= max_length:
            base = base[:, 0:max_length]
        # src:bm_size*原句长度
        # feed:bm_size*各自翻译长度

        repeat = 1
        complete = []
        complete_scores = []
        working_candidates = beam_size
        vocal = len(zh_vocab)

        candidates = torch.full((beam_size, 1), 0).to(device)
        candidates_scores = torch.zeros_like(candidates)

        while working_candidates != 0:
            if repeat == 1:
                base_feed = torch.tensor([[0]]).to(device)
                output = model(base, base_feed)
            else:
                src = base.repeat(working_candidates, 1)
                output = model(src, candidates)

            predict = output[:, -1]  # W*Vocal
            res = torch.log_softmax(predict, 1)  # W*Vocal

            if repeat == 1:
                sum_res = res
            else:
                sum_res = candidates_scores + res

            # 展平求topK
            sum_res = sum_res.reshape(1, -1)  # 1,W*Vocal
            relative_index = torch.topk(k=working_candidates, input=sum_res, dim=-1)  # 1*W

            next_candidates = []
            next_scores = []

            for ele in relative_index.indices.squeeze(0):
                real_pos = torch.tensor([int(ele % vocal)], device=device)
                pre_index = int(ele / vocal)
                score = sum_res[0][ele]

                pre_serial = candidates[pre_index]
                next_serial = torch.cat((pre_serial, real_pos))

                if real_pos == 1 or repeat == max_length:
                    working_candidates -= 1
                    complete.append(next_serial)
                    complete_scores.append(score)
                else:
                    next_candidates.append(next_serial)
                    next_scores.append(score)

            if working_candidates == 0:
                break
            candidates = torch.stack(next_candidates, 0)
            candidates_scores = torch.stack(next_scores, 0).unsqueeze(1)
            repeat += 1

        complete_scores = torch.stack(complete_scores, 0)

        index = 0
        for ele in complete:
            complete_scores[index] /= ele.shape[0]
            index += 1

        ans_index = torch.argmax(complete_scores)
        ans = complete[ans_index]

        tgt = ''.join(zh_vocab.lookup_tokens(ans.tolist())).replace("<bos>", "").replace("<eos>", "")
        return tgt


while True:
    user_input = input("请输入英文原句:")
    size = input("请输入搜索尺寸:")
    res = translate_bs(user_input, int(size))
    print(res)
