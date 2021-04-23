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
                          batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.fc = nn.Linear(enc_hid_size * 2, dec_hid_size)
        else:
            self.fc = nn.Linear(enc_hid_size, dec_hid_size)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout_rnn)
    def forward(self, x):
        sent_output, sent_hidden = self.rnn(x, None)
        local_representation = self.pool(sent_output.transpose(1, 2)).squeeze(-1)
        return self.dropout(local_representation)

class generate(nn.Module):
    def __init__(self, args):
        super(generate, self).__init__()

        self.embedding = get_embedding(args)
        self.share_layer = GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     args.bidirectional, args.dec_hid_size,
                                                     args.dropout_rnn)
        self.private_layer = nn.ModuleList([GRU_attn(args.glove_dim, args.enc_hid_size, args.rnn_layers,
                                                     args.bidirectional, args.dec_hid_size,
                                                     args.dropout_rnn)]*args.task_num)

        if args.bidirectional:
            self.fc_target = nn.Linear(args.enc_hid_size*4, args.output_size)
            self.fc_task = nn.Linear(args.enc_hid_size*2, args.task_num)
        else:
            self.fc_target = nn.Linear(args.enc_hid_size*2, args.output_size)
            self.fc_task = nn.Linear(args.enc_hid_size, args.task_num)

    def forward(self, input):

        emb = self.embedding(input["x"])
        share_layer = self.share_layer(emb)
        private_layer = self.private_layer[input["task_id"]](emb)
        fusion_layer = torch.cat((share_layer, private_layer), dim=1)
        target = self.fc_target(fusion_layer)

        return [target, None, share_layer]

class Discriminator(nn.Module):

    def __init__(self, dim, num_tasks):
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(dim, num_tasks)

    def forward(self, input):
        return self.linear(input)
