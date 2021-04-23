import torch
import torch.nn as nn
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

class GRU_attn(nn.Module):
    """
    GRU
    """
    def __init__(self, glove_dim, enc_hid_size, rnn_layers, bidirectional, dec_hid_size, dropout_rnn, device="cuda"):
        super(GRU_attn, self).__init__()
        self.device = device
        self.rnn = nn.LSTM(glove_dim, enc_hid_size, rnn_layers,
                          batch_first=True, bidirectional=False)

        self.pool = nn.AdaptiveMaxPool1d(1)
    def forward(self, x):
        sent_output, sent_hidden = self.rnn(x, None)
        local_representation = self.pool(sent_output.transpose(1, 2)).squeeze(-1)
        return local_representation



class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()

        self.embedding = get_embedding(args)
        self.private_layer = GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     args.bidirectional, args.dec_hid_size,
                                                     args.dropout_rnn)

        self.fc_target = nn.Linear(args.enc_hid_size, args.output_size)
        nn.init.xavier_normal_(self.fc_target.weight)


        self.dropout = nn.Dropout(args.dropout)
    def forward(self, input):

        emb = self.embedding(input["x"])
        private_layer = self.dropout(self.private_layer(emb))
        target = self.fc_target(private_layer)

        return [target, None, None]
