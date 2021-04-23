class Config():

    def __init__(self, task, model, glove_dim=300):
        self.data_path = "../data/"
        self.model = model
        self.seed = 99
        self.task = task
        self.task_num = len(task)
        self.bert_path = "../data/bert/"
        self.bert_dim = 768
        self.segmentation = True
        self.max_len = 420
        self.batch_size = 100
        self.taskids = {}
        self.id_task = {}
        self.drop_last_batch = False
        self.workers = 0
        self.device = "cuda"
        self.use_gpu = True
        self.num_channels = [64, 64, 64]
        self.kernel_sizes = [3, 4, 5]
        self.num_filters = 2
        self.hidden_size = 300
        self.out_size = 32
        self.pool_type = "max"
        self.lr = 0.2
        self.num_epochs = 5
        self.dropout = 0.5
        self.dropout_rnn = 0.5
        self.dropout_trans = 0.5

        if model == "task_recognition":
            self.model_path = "./save_checkpoint/task_recognition/"
            self.resume_path = None
        elif model == "MT_GRU":
            self.model_path = "./save_checkpoint/MT_GRU/"
            self.resume_path = None
        elif model == "MT_CNN":
            self.model_path = "./save_checkpoint/MT_CNN/"
            self.resume_path = None
        elif model == "MTL_ASP":
            self.model_path = "./save_checkpoint/MTL_ASP/"
            self.resume_path = None
        elif model == "CNN":
            self.model_path = "./save_checkpoint/CNN/"
            self.resume_path = None
        elif model == "LSTM":
            self.model_path = "./save_checkpoint/LSTM/"
            self.resume_path = None
        elif model == "Bi-LSTM":
            self.model_path = "./save_checkpoint/Bi_LSTM/"
            self.resume_path = None
        elif model == "LSTM_Att":
            self.model_path = "./save_checkpoint/LSTM_Att/"
            self.resume_path = None
        elif model == "no_BERT_MTL":
            self.model_path = "./save_checkpoint/no_BERT_MTL/"
            self.resume_path = None
        elif model == "no_task_MTL":
            self.model_path = "./save_checkpoint/no_task_MTL/"
            self.resume_path = None

        self.result_path = "./result/"

        if glove_dim == 300:
            self.vocab_size = 18766
            self.glove_dim = 300
            self.glove_file = "../data/glove/glove_300d.npy"
            self.word2id_file = "../data/glove/word2id.npy"
        elif glove_dim == 200:
            self.vocab_size = 400000
            self.glove_dim = 200
            self.glove_file = "../data/glove/glove_200d.npy"
            self.word2id_file = "../data/glove/word2id_40.npy"
        elif glove_dim == 100:
            self.vocab_size = 400000
            self.glove_dim = 100
            self.glove_file = "../data/glove/glove_100d.npy"
            self.word2id_file = "../data/glove/word2id_40.npy"
        elif glove_dim == 50:
            self.vocab_size = 400000
            self.glove_dim = 50
            self.glove_file = "../data/glove/glove_50d.npy"
            self.word2id_file = "../data/glove/word2id_40.npy"

        self.min_emb = 100
        self.num_heads = 4
        self.weight_decay = 0.1
        self.max_norm = 0.9

        self.ndim = 256
        self.nhead = 4
        self.nhid = 64
        self.nlayers = 6

        self.target_weight = 1
        self.task_weight = 0.05
        self.diff_weight = 0.01

        self.enc_hid_size = 64
        self.dec_hid_size = 64
        self.num_classes = 2
        self.num_directions = 1
        self.rnn_layers = 1
        self.output_size = 2

        self.alpha = 1
        self.bidirectional = True

