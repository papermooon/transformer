from torch.utils.data import DataLoader
from hyperPara import *
from process import CommentaryDataset, en_tokens, zh_tokens, commentary_collate_fn

dataset = CommentaryDataset(en_tokens, zh_tokens)
data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=commentary_collate_fn)
