import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import pandas as pd
import os

class Train(object):

    def __init__(self, args, epochs, optimizer, optimizer_d, scheduler):
        self.args = args
        self.epoch = epochs
        self.optimizer = optimizer
        self.optimizer_d = optimizer_d
        self.scheduler = scheduler

        self.train_acc = {"target": {"all": {"acc": 0, "f1": 0}}}
        self.best_acc = {"target": {"acc": 0, "f1": 0}}
        self.test_acc = {"target_acc": {"all": 0}, "target_f1": {"all": 0}}

        self.result_path = args.result_path

    def fit(self, model, discriminator, train_loader, test_loader):

        for epoch in range(self.epoch):
            loss = self.train_epoch(model, discriminator, train_loader)

            self.test_epoch(model, test_loader)
            if self.best_acc["target"]["acc"] < self.test_acc["target_acc"]["all"] and epoch > 0:
                self.save_checkpoint(model, epoch)

            print("epoch: {} loss: {}\n train_acc: {}\n test_acc:{}".format(epoch, loss, self.train_acc, self.test_acc))

    def train_epoch(self, model, discriminator, train_loader):
        pred, true, losses = {"target": [], "task": []}, {"target": [], "task": []}, \
                           {"target": [], "task": [], "dis_loss": [], "dis_loss0": [], "total": []}
        for x, label, task_id, seq_len in tqdm(train_loader):

            targets = torch.LongTensor([label, task_id]).cuda()

            outputs = model({"x": x, "task_id": task_id[0], "seq_len": seq_len})
            target_loss = F.cross_entropy(outputs[0], targets[0]) * self.args.target_weight
            if not outputs[1]==None:
                task_loss = F.cross_entropy(outputs[1], targets[1]) * self.args.task_weight
                total_loss = target_loss + task_loss
            else:
                total_loss = target_loss
            if not outputs[2] == None:

                dis_loss = F.cross_entropy(discriminator(outputs[2]), targets[1]) * self.args.diff_weight
                self.optimizer_d.zero_grad()
                dis_loss.backward(retain_graph=True)
                self.optimizer_d.step()
                total_loss += dis_loss


            self.optimizer.zero_grad()
            total_loss.backward()
            clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), self.args.max_norm)
            self.optimizer.step()
            self.scheduler.step()

            losses["target"].append(target_loss.tolist())
            losses["total"].append(total_loss.tolist())

            pred["target"].extend(outputs[0].argmax(dim=1).tolist())
            true["target"].extend(label)

        self.train_acc["target"] = accuracy_score(pred["target"], true["target"])
        torch.cuda.empty_cache()


        return {"target": np.mean(losses["target"]), "total": np.mean(losses["total"])}

    def test_epoch(self, model, test_loader):
        model.eval()
        pred, true, results = {"target": {}, "task": {}}, {"target": {}, "task": {}}, {"target": {"all": {}}, "task": {"all": {}}}
        for task in self.args.task:
            pred["target"][task] = []
            true["target"][task] = []
        with torch.no_grad():
            for x, label, task_id, seq_len in tqdm(test_loader):
                result = model({"x": x, "task_id": task_id[0], "seq_len": seq_len})
                pred["target"][self.args.id_task[task_id[0]]].extend(result[0].argmax(dim=1).tolist())
                true["target"][self.args.id_task[task_id[0]]].extend(label)
        self.test_acc = self.evaluate(true, pred, self.args.task)

        model.train()
        torch.cuda.empty_cache()

    def save_checkpoint(self, model, epoch):
        self.best_acc["target"]["acc"] = self.test_acc["target_acc"]["all"]
        state = {'model': model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch + 1}
        if not os.path.exists(self.args.model_path):
            os.mkdir(self.args.model_path)

        model_name = self.args.model_path + self.args.model + '_' + str(self.args.seed) +\
                     '_' + str(epoch + 1) + '_' + str(format(self.best_acc["target"]["acc"], '.3f')) + '.pth'
        print("model save in {}".format(model_name))
        try:
            torch.save(state, model_name)
        except:
            print("Failed to save file")
        result_path = self.result_path + str(format(self.best_acc["target"]["acc"], '.3f')) + "_" + self.args.model_path.split("/")[-2] + ".csv"
        print(result_path)
        pd.DataFrame(self.test_acc).to_csv(result_path)

    def evaluate(self, true, pred, tasks):
        results = {"target_acc": {"all": 0}, "target_f1": {"all": 0}, "task_acc": {"all": 0}, "task_f1": {"all": 0}}
        for task in tasks:
            results["target_acc"][task] = 0
            results["target_f1"][task] = 0

        all_acc, all_f1 = {"target": [], "task": []}, {"target": [], "task": []}
        for task, true_value, pred_value in zip(true["target"].keys(), true["target"].values(),
                                                pred["target"].values()):
            results["target_acc"][task] = accuracy_score(true_value, pred_value).round(3)
            results["target_f1"][task] = f1_score(true_value, pred_value).round(3)
            all_acc["target"].append(results["target_acc"][task])
            all_f1["target"].append(results["target_f1"][task])

        results["target_acc"]["all"] = np.mean(all_acc["target"]).round(3)
        results["target_f1"]["all"] = np.mean(all_f1["target"]).round(3)

        return results


    def resume(self, model, model_path, test_loader, args):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        pred, true, results = {"target": {}, "task": {}}, {"target": {}, "task": {}}, {"target": {"all": {}}, "task": {"all": {}}}
        for task in args.task:
            pred["target"][task] = []
            true["target"][task] = []

        model.eval()
        with torch.no_grad():
            for x, label, task_id, seq_len in tqdm(test_loader):
                result = model({"x": x, "task_id": task_id[0], "seq_len": seq_len})
                pred["target"][args.id_task[task_id[0]]].extend(result[0].argmax(dim=1).tolist())
                true["target"][args.id_task[task_id[0]]].extend(label)

        results = self.evaluate(true, pred, args.task)
        print(results)

