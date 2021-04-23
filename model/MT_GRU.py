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

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        src_len = enc_output.shape[1]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class GRU_attn(nn.Module):
    """
    GRU
    """
    def __init__(self, glove_dim, enc_hid_size, rnn_layers, bidirectional, dec_hid_size, dropout_rnn, device="cuda"):
        super(GRU_attn, self).__init__()
        self.device = device
        self.rnn = nn.GRU(glove_dim, enc_hid_size, rnn_layers,
                          batch_first=True, bidirectional=bidirectional)

        if bidirectional:
            self.fc = nn.Linear(enc_hid_size * 2, dec_hid_size)
        else:
            self.fc = nn.Linear(enc_hid_size, dec_hid_size)
        self.attn = Attention(enc_hid_size, dec_hid_size)
        self.dropout = nn.Dropout(dropout_rnn)
    def forward(self, x):
        sent_output, sent_hidden = self.rnn(x)
        s = torch.tanh(self.fc(torch.cat((sent_hidden[-2, :, :], sent_hidden[-1, :, :]), dim=1)))
        attn_weights = self.attn(s, sent_output)
        local_representation = torch.bmm(sent_output.transpose(1, 2), attn_weights.unsqueeze(2)).squeeze(-1)

        return local_representation

class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()

        self.embedding = get_embedding(args)
        self.private_layer = nn.ModuleList([GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     args.bidirectional, args.dec_hid_size,
                                                     args.dropout_rnn)]*args.task_num)
        if args.bidirectional:
            self.fc_target = nn.Linear(args.enc_hid_size*2, args.output_size)
        else:
            self.fc_target = nn.Linear(args.enc_hid_size, args.output_size)
        nn.init.xavier_normal_(self.fc_target.weight)
        self.dropout = nn.Dropout(args.dropout)
    def forward(self, input):

        emb = self.embedding(input["x"])
        private_layer = self.dropout(self.private_layer[input["task_id"]](emb))
        target = self.fc_target(private_layer)

        return [target, None, None]
