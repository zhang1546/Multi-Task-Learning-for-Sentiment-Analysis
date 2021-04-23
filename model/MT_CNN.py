import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class get_embedding(nn.Module):

    def __init__(self, args):
        super(get_embedding, self).__init__()
        self.args = args
        self.init_glove()
        self.word_dim = args.glove_dim

    def forward(self, x):
        return self.get_glove(x)

    def init_glove(self):
        """
        load the GloVe model
        """
        self.word2id = np.load(self.args.word2id_file, allow_pickle=True).tolist()
        self.glove = nn.Embedding(self.args.vocab_size, self.args.glove_dim)
        emb = torch.from_numpy(np.load(self.args.glove_file, allow_pickle=True)).to(self.args.device)
        self.glove.weight.data.copy_(emb)
        self.word_dim = self.args.glove_dim
        self.glove.weight.requires_grad = False

    def get_glove(self, sentence_lists):
        """
        get the glove word embedding vectors for a sentences
        """
        max_len = max(map(lambda x: len(x), sentence_lists))
        sentence_lists = list(map(lambda x: list(map(lambda w: self.word2id.get(w, 0), x)), sentence_lists))
        sentence_lists = list(map(lambda x: x + [self.args.vocab_size - 1] * (max_len - len(x)), sentence_lists))
        sentence_lists = torch.LongTensor(sentence_lists).to(self.args.device)

        return self.glove(sentence_lists)


class CNN_layers(nn.Module):

    def __init__(self, num_channels, kernel_sizes, glove_dim, device="cuda"):
        super(CNN_layers, self).__init__()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=glove_dim,
                                        out_channels=c,
                                        kernel_size=k).to(device))
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight.data)
            nn.init.uniform_(conv.bias, 0, 0)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = torch.cat([
            self.pool(F.gelu(conv(x))).squeeze(-1) for conv in self.convs], dim=1)
        return x


class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()
        self.embedding = get_embedding(args)
        self.private_layer = nn.ModuleList(
            [CNN_layers(args.num_channels, args.kernel_sizes, args.glove_dim)] * args.task_num)
        self.fc_target = nn.Linear(sum(args.num_channels), args.output_size)
        nn.init.xavier_normal_(self.fc_target.weight)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, input):
        emb = self.embedding(input["x"])
        share_layer = self.private_layer[input["task_id"]](emb.permute(0, 2, 1))
        target = self.fc_target(share_layer)

        return [target, None, None]
