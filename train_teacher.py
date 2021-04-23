import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from util import setup_seed
import os


class Train(object):
    def __init__(self, args, teacher_model, student_model, train_loader, test_loader, optimizer, loss, scheduler,
                 seed=0, path=None):
        self.args = args
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = args.device
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        self.best_result = {"seed": 0, "epoch": 0, "acc": 0.0, "f1": 0.0}
        self.seed = seed
        # self.print_train = "epoch: {} loss: {:.3f}, lr: {:.5f}, acc: {:.3f}, f1: {:.3f}"
        self.epoch = 0
        self.path = path
        self.loss_t = torch.nn.MSELoss()  # 先试试，可以试试KLDivLoss()
        self.best_loss = float("inf")
        self.alpha = args.alpha

    def train_teacher(self):
        for self.seed in range(self.args.seed):
            setup_seed(self.seed)
            for epoch in range(self.args.num_epochs):
                self.teacher_model.train()
                self.train(epoch, self.teacher_model, self.loss, self.optimizer, self.scheduler,
                           self.train_loader, self.device)
                result, _ = self.evaluate(self.teacher_model, self.test_loader, self.device)

                if self.best_result["acc"] < result["acc"]:
                    self.best_result["acc"] = result["acc"]
                    self.best_result["f1"] = result["f1"]
                    self.best_result["seed"] = self.seed
                    if epoch > 1:
                        state = {'model': self.teacher_model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                                 'epoch': epoch + 1}
                        path = self.args.model_path + 'teacher_seed' + str(self.seed) + '_epoch' + str(
                            epoch + 1) + '_acc' + str(
                            format(self.best_result["acc"], '.3f')) + '.pth'
                        torch.save(state, path)
                        print("success save model: ", os.path.abspath(path))
            # setup_seed(self.seed)
            # for self.epoch in range(self.args.num_epochs):
            #     y_true, y_pred, lr_list, train_l_sum, batch_count = [], [], [], 0.0, 0.0
            #     for x, y in tqdm(self.train_loader):
            #         label, task_id = list(zip(*y))
            #         label = torch.LongTensor(label).to(self.device)
            #         y_score = self.teacher_model(x)
            #         l = self.loss(y_score, label)
            #
            #         self.optimizer.zero_grad()
            #         l.backward()
            #         self.optimizer.step()
            #
            #         train_l_sum += l.cpu().item()
            #         batch_count += 1
            #         y_true.extend(label.cpu().tolist())
            #         y_pred.extend(y_score.argmax(dim=1).cpu().tolist())
            #     torch.cuda.empty_cache()
            #     lr_list.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            #     self.scheduler.step()
            #
            #     print(self.print_train.format(self.epoch, train_l_sum / batch_count,
            #                                    self.optimizer.state_dict()['param_groups'][0]['lr'],
            #                                     accuracy_score(y_true, y_pred),
            #                                    f1_score(y_true, y_pred)))
            #     result = self.evaluate(self.teacher_model, self.test_loader, self.device)
            #
            #     self.teacher_model.train()
            #     if self.best_result["acc"] < result["acc"]:
            #         self.best_result["acc"] = result["acc"]
            #         self.best_result["f1"] = result["f1"]
            #         self.best_result["seed"] = self.seed
            #         ## 模型teachermodel
            #         if self.epoch > 1:
            #             state = {'model': self.teacher_model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            #                      'epoch': self.epoch + 1}
            #             path = self.args.model_path + 'teacher_seed' + str(self.seed) + '_epoch' + str(
            #                 self.epoch + 1) + '_acc' + str(
            #                 format(self.best_result["acc"], '.3f')) + '.pth'
            #             torch.save(state, path)
            #             print("success save model: ", os.path.abspath(path))

    def train_student(self):
        for self.seed in range(self.args.seed):
            setup_seed(self.seed)
            for epoch in range(self.args.num_epochs):
                self.student_model.train()
                self.train(epoch, self.student_model, self.loss, self.optimizer, self.scheduler,
                           self.train_loader, self.device)
                self.evaluate(self.student_model, self.test_loader, self.device)
            #
            # if self.best_result["acc"] < result["acc"]:
            #     self.best_result["acc"] = result["acc"]
            #     self.best_result["f1"] = result["f1"]
            #     self.best_result["seed"] = self.seed
            #     if self.epoch > 1:
            #         state = {'model': self.teacher_model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            #                  'epoch': self.epoch + 1}
            #         path = self.args.model_path + 'student_seed' + str(self.seed) + '_epoch' + str(
            #             self.epoch + 1) + '_acc' + str(
            #             format(self.best_result["acc"], '.3f')) + '.pth'
            #         torch.save(state, path)
            #         print("success save model: ", os.path.abspath(path))

    def distilling(self):
        self.recurrent(self.path, self.teacher_model, None, eval=False)
        _, teacher_train = self.evaluate(self.teacher_model, self.train_loader, self.device)
        _, teacher_test = self.evaluate(self.teacher_model, self.test_loader, self.device)

        for epoch in range(self.args.num_epochs):
            # 训练
            self.student_model.train()
            y_pred, y_true, train_l_sum, batch_count = [], [], 0, 0
            for x, y in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                label, task_id = list(zip(*y))
                label = torch.LongTensor(label).to(self.device)
                y_student = self.student_model(x)
                loss = self.get_loss(teacher_train[batch_count], y_student, label, self.alpha, (1-self.alpha))
                loss.backward()
                self.optimizer.step()
                y_pred.extend(y_student.argmax(dim=1).cpu().tolist())
                y_true.extend(label.cpu().tolist())
                train_l_sum += loss.cpu().item()
                batch_count += 1
            torch.cuda.empty_cache()
            self.scheduler.step()
            print("epoch: {} loss: {:.3f}, lr: {:.5f}, acc: {:.3f}, f1: {:.3f}"
                  .format(epoch, train_l_sum / batch_count, self.optimizer.state_dict()['param_groups'][0]['lr'],
                          accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)))
            # 测试
            self.student_model.eval()
            y_true, y_pred, loss_total, i = [], [], 0, 0
            for x, y in tqdm(self.test_loader):
                label, task_id = list(zip(*y))
                label = torch.LongTensor(label).to(self.device)
                # y = self.teacher_model.predict(x).cpu()
                y_true.extend(label.cpu().tolist())
                # y_pred.extend(model.predict(x).cpu().tolist())
                y_score = self.student_model(x)
                loss = self.get_loss(teacher_test[i], y_score, label, self.alpha, (1-self.alpha))
                loss_total += loss.cpu()
                # y_scores.append(y_score.cpu())
                y_pred.extend(y_score.argmax(dim=1).cpu().tolist())
                i += 1
            test_loss = loss_total/i
            torch.cuda.empty_cache()
            print("Val Loss: {:.3f}, acc: {:.3f} ".format(test_loss, accuracy_score(y_true, y_pred)))
            if test_loss < self.best_loss:
                self.best_loss = test_loss
                print("best epoch: {}, loss: {:.3f}, acc: {:.3f}".format(epoch, test_loss, accuracy_score(y_true, y_pred)))



    @staticmethod
    def train(epoch, model, loss, optimizer, scheduler, data_loader, device):
        # for self.seed in range(seed):
        # setup_seed(seed)
        # for epoch in range(num_epochs):
        y_true, y_pred, lr_list, train_l_sum, batch_count = [], [], [], 0.0, 0.0
        for x, y in tqdm(data_loader):
            label, task_id = list(zip(*y))
            label = torch.LongTensor(label).to(device)
            y_score = model(x)
            l = loss(y_score, label)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            train_l_sum += l.cpu().item()
            batch_count += 1
            y_true.extend(label.cpu().tolist())
            y_pred.extend(y_score.argmax(dim=1).cpu().tolist())
        torch.cuda.empty_cache()
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()

        print("epoch: {} loss: {:.3f}, lr: {:.5f}, acc: {:.3f}, f1: {:.3f}"
              .format(epoch, train_l_sum / batch_count, optimizer.state_dict()['param_groups'][0]['lr'],
                      accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)))
        # result = self.evaluate(teacher_model, self.test_loader, self.device)

        # return

    @staticmethod
    def evaluate(model, data_loader, device):

        model.eval()
        y_true, y_pred, y_scores = [], [], []
        with torch.no_grad():
            for x, y in data_loader:
                label, task_id = list(zip(*y))
                label = torch.LongTensor(label).to(device)
                # y = self.teacher_model.predict(x).cpu()
                y_true.extend(label.cpu().tolist())
                # y_pred.extend(model.predict(x).cpu().tolist())
                y_score = model(x).cpu()
                y_scores.append(y_score)
                y_pred.extend(y_score.argmax(dim=1).tolist())
            torch.cuda.empty_cache()
        result = {"acc": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
        print("evaluate acc: {:.3f}, f1: {:.3f}".format(result["acc"], result["f1"]))
        return result, y_scores

    @staticmethod
    def recurrent(path, model, data_loader, eval=True, device="cuda"):

        setup_seed(int(path.split("_")[2][-1]))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        if eval:
            Train.evaluate(model, data_loader, device)
        #
        # y_true, y_pred = [], []
        # model.eval()
        # with torch.no_grad():
        #     for x, y in data_loader:
        #         label, taskid = list(zip(*y))
        #         label = torch.LongTensor(label).to(device)
        #         # y = teachermodel.predict(x).cpu()
        #         y_true.extend(label.cpu().tolist())
        #         y_pred.extend(model.predict(x).cpu().tolist())
        #         torch.cuda.empty_cache()
        #         # gpu_tracker.track()
        # result = {"acc": accuracy_score(y_true, y_pred), "f1": f1_score(y_true, y_pred)}
        # print("evaluate acc: {:.3f}, f1: {:.3f}".format(result["acc"], result["f1"]))

    @staticmethod
    def get_loss(t_logits, s_logits, label, a, T):  # a = 1, T = 0
        loss1 = torch.nn.CrossEntropyLoss()
        loss2 = torch.nn.MSELoss()
        # loss2 = torch.nn.KLDivLoss()
        if t_logits.is_cuda != s_logits.is_cuda:
            t_logits = t_logits.cuda()
        loss = a * loss1(s_logits, label) + T * loss2(t_logits, s_logits)
        # print(loss1(s_logits, label),loss2(t_logits, s_logits))
        return loss

    def calculate_single_task_score(self, result_dict):
        """return {"task": [acc, f1]}"""
        df = pd.DataFrame(result_dict)
        result = {}
        for task in self.task_ids.keys():
            y_true, y_pred = df[df["y_label"] == task]["y_true"].to_list(), df[df["y_label"] == task][
                "y_pred"].to_list()
            result[task] = {"acc": "%.3f" % accuracy_score(y_true, y_pred),
                            "f1": "%.3f" % f1_score(y_true, y_pred, average='binary')}
        result["all"] = {"acc": "%.3f" % accuracy_score(result_dict["y_true"], result_dict["y_pred"]),
                         "f1": "%.3f" % f1_score(result_dict["y_true"], result_dict["y_pred"], average='binary')}
        return result
