import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel, AutoTokenizer

class BERT(nn.Module):
    def __init__(self, args):
        super(BERT, self).__init__()
        self.args = args
        self.bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
        self.bert = AutoModel.from_pretrained(args.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = False


    def forward(self, x):
        word_emb = self.get_embedding(x)
        return word_emb

    def get_embedding(self, sentence_lists):
        sentence_lists = [' '.join(x) for x in sentence_lists]
        ids = self.bert_tokenizer(sentence_lists, padding=True, return_tensors="pt")
        inputs = ids['input_ids']
        if self.args.use_gpu:
            inputs = inputs.to(self.args.device)
        return self.bert(inputs)[0]


class CNN_layers(nn.Module):
    """
    CNN
    """
    def __init__(self, num_channels, kernel_sizes, glove_dim, device="cuda"):
        super(CNN_layers, self).__init__()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=glove_dim,
                                        out_channels=c,
                                        kernel_size=k).to(device))
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight.data)
            nn.init.uniform_(conv.bias, 0, 0)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = torch.cat([
            self.pool(F.relu(conv(x))).squeeze(-1) for conv in self.convs], dim=1)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=450):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()
        self.embedding = BERT(args)
        self.task_specific = nn.ModuleList([CNN_layers(args.num_channels, args.kernel_sizes, args.bert_dim)]*args.task_num)

        if args.bidirectional:
            self.fc_specific = nn.Linear(sum(args.num_channels), args.output_size)

        else:
            self.fc_specific = nn.Linear(sum(args.num_channels)+args.enc_hid_size, args.output_size)
        nn.init.xavier_normal_(self.fc_specific.weight)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input):
        emb = self.embedding(input["x"])
        specific = self.task_specific[input["task_id"]](emb.permute(0, 2, 1))
        target = self.fc_specific(specific)

        return [target, None, None]
