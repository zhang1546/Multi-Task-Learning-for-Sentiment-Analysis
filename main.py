from config import Config
from data import prepare_data
from util import setup_seed
import torch.optim.lr_scheduler as lr_scheduler
from ranger import Ranger
from train import Train
import time
# dataset 1
# args = Config(task=["books", "electronics", "dvd", "kitchen_housewares", "apparel",
#                     "camera_photo", "health_personal_care", "music", "toys_games", "video",
#                     "baby", "magazines", "software", "sports_outdoors", "imdb", "MR"],
#               model="MTL_ASP", glove_dim=300)
# dataset 2
args = Config(task=["daily_necessities", "literature", "entertainment", "media"],
              model="no_task_MTL", glove_dim=300)

print(args.taskids)
train_loader, test_loader = prepare_data(args)

discriminator = None
optimizer_d = None
if args.model == "task_recognition":
    from model.task_recognition import generate
elif args.model == "MTL_ASP":
    from model.MTL_ASP import generate, Discriminator
    if args.bidirectional:
        discriminator = Discriminator(args.enc_hid_size * 2, args.task_num).cuda()
    else:
        discriminator = Discriminator(args.enc_hid_size, args.task_num).cuda()
    optimizer_d = Ranger(discriminator.parameters())
elif args.model == "MT_GRU":
    from model.MT_GRU import generate
elif args.model == "MT_CNN":
    from model.MT_CNN import generate
elif args.model == "CNN":
    from model.CNN import generate
elif args.model == "LSTM":
    from model.LSTM import generate
elif args.model == "Bi-LSTM":
    from model.Bi_LSTM import generate
elif args.model == "LSTM_Att":
    from model.LSTM_Att import generate
elif args.model == "no_BERT_MTL":
    from model.no_BERT_MTL import generate
elif args.model == "no_task_MTL":
    from model.no_task_MTL import generate

setup_seed(args.seed)
model = generate(args).to(args.device)
optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()),
                   lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
train = Train(args, args.num_epochs, optimizer, optimizer_d, scheduler)

## 测试summary
from model.summary import summary
for x, label, task_id, seq_len in train_loader:
    break
summary(model, {"x": x, "task_id": task_id[0], "seq_len": seq_len}, (100, 400))

start = time.time()
train.fit(model, discriminator, train_loader, test_loader)
end = time.time()

print("\n\nAll time is ", end-start)

# train.resume(model, args.resume_path, test_loader, args)
