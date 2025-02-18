import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import os

from    learner import Learner
from    copy import deepcopy
import math
import sklearn.metrics

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """
        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry,rms_temp):
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        y_spt = y_spt.to(torch.int64)
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            rms=rms_temp[i]
            w1 = (math.exp(rms[0]) + math.exp(rms[1]) + math.exp(rms[2])) / math.exp(rms[0])
            w2 = (math.exp(rms[0]) + math.exp(rms[1]) + math.exp(rms[2])) / math.exp(rms[1])
            w3 = (math.exp(rms[0]) + math.exp(rms[1]) + math.exp(rms[2])) / math.exp(rms[2])
            weight = torch.tensor([w1, w2, w3], device='cuda:0')
            # weight = torch.tensor([w1, w2, w3])
            # y_spt[i] =  y_spt[i].to(torch.int64)
            loss = F.cross_entropy(logits, y_spt[i])
            # target=y_spt[i]
            # input = F.softmax(logits, dim=1)
            # target=y_spt[i]
            # log_soft_out = torch.log(input)
            # # rms=
            # weight=torch.tensor([0.3,0.5,0.2],device='cuda:0')
            # loss_c = F.nll_loss(log_soft_out, target,weight=weight)
            #
            # loss_c = 0.0
            # for b in range(target.shape[0]):
            #     for i in range(target.shape[1]):
            #         for j in range(target.shape[2]):
            #             loss_c -= torch.log(input[b][target[b][i][j]][i][j])
            # # 求均值
            # c=loss_c / 9
            # print(loss / 9)


            grad = torch.autograd.grad(loss, self.net.parameters())
            # parameters=self.net.parameters()
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) #更新参数
            #zip(grad, self.net.parameters()) 作用是把grad和self.net.parameters()组成一个元组
            # [(grad[0],self.net.parameters()[0]),(grad[1],self.net.parameters()[2]),...]
            # map 映射表示 某种映射关系

            # this is the loss and accuracy before first update
            y_qry = y_qry.to(torch.int64)
            with torch.no_grad(): #之后的内容不进行计算图构建 也就是不用跟踪反向梯度计算
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item() #若相同位置的两个元素相同，则返回True；若不同，返回False。
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.

                loss_q = F.cross_entropy(logits_q, y_qry[i],weight)
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct


            # # w1=rms[0]/(rms[0]+rms[1]+rms[2])

            # loss = F.cross_entropy(logits, y_spt[i],weight)


        # end of all tasks
        # sum over all losses on query set across all tasks
        cc=losses_q[-1]
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs


    # def finetunning(self, x_spt, y_spt, x_qry, y_qry):
    def finetunning(self, x_spt, y_spt, x_qry, y_qry, TorF):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        # print("load the trained net")
        # weights_dict = torch.load('./net.pth')
        # print("load done")
        # model = deepcopy(self.net)
        # load_weights_dict = {k: v for k, v in weights_dict.items() if model.state_dict()[k].numel() == v.numel()}
        # model.load_state_dict(load_weights_dict, strict=False)
        # output = model(x_qry)
        # pred = F.softmax(output, dim=1).argmax(dim=1)
        #
        # y_qry_test = y_qry.to(torch.int64)
        # print('loaded_pred',pred)
        # print('loaded_label',y_qry_test)




        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]  # (0, 10)

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # print("load the trained net")
        # weights_dict = torch.load('./net.pth')
        # load_weights_dict = {k: v for k, v in weights_dict.items() if net.state_dict()[k].numel() == v.numel()}
        # net.load_state_dict(load_weights_dict, strict=False)




        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        y_spt = y_spt.to(torch.int64)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
        accc=[]
        precision=[]
        F1=[]
        Recall=[]
        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)

            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # len(logits_q) = 30  --> tensor([[-0.2803, -1.3926, 0.9923], [-0.6793, -2.7332, 0.6125],```])
            # print('logits_q',logits_q)
            # loss_q will be overwritten and just keep the loss_q on last update step.

            y_qry = y_qry.to(torch.int64)
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                # print('code_result',pred_q,y_qry)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        current_file_path = os.path.dirname(os.path.abspath(__file__))
        net_save_path = f"{current_file_path}/fastweight.pth"
        # net_vars_save_path = f"{current_file_path}/net_vars.pth"
        # net_vars_bn_save_path = f"{current_file_path}/net_vars_bn.pth"
        # torch.save(fast_weights, net_save_path)
        # torch.save(net.vars, net_vars_save_path)
        # torch.save(net.vars_bn, net_vars_bn_save_path)
        # fw = torch.load("fastweight.pth")
        fw = fast_weights
        net_test = deepcopy(self.net)
        pred_ture = net_test(x_qry, fw, bn_training=True)
        pred_false = net_test(x_qry, fw, bn_training=False)
        with torch.no_grad():
            pred_q_true = F.softmax(pred_ture, dim=1).argmax(dim=1)
            # print('test_result_true',pred_q_true)
            pred_q_false = F.softmax(pred_false, dim=1).argmax(dim=1)
            # print('test_result_false',pred_q_false)


        del net
        y_te = y_qry.cpu().numpy()
        predict = pred_q.cpu().numpy()
        acc = np.array(correct) / querysz
        # micro_precision = sklearn.metrics.precision_score(y_te, predict, labels=None, average='micro',
        #                                                   sample_weight=None)
        macro_precision = sklearn.metrics.precision_score(y_te, predict, labels=None, average='macro',
                                                          sample_weight=None)
        # micro_f1 = sklearn.metrics.f1_score(y_te, predict, labels=None, average='micro',
        #                                     sample_weight=None)
        macro_f1 = sklearn.metrics.f1_score(y_te, predict, labels=None, average='macro',
                                            sample_weight=None)
        # micro_recall = sklearn.metrics.recall_score(y_te, predict, labels=None, average='micro',
        #                                             sample_weight=None)
        macro_recall = sklearn.metrics.recall_score(y_te, predict, labels=None, average='macro',
                                                    sample_weight=None)
        accs = np.array(corrects) / querysz

        return accs,acc,macro_precision,macro_f1,macro_recall


class Meta_base(nn.Module):
    """
    Meta Learner  Base 版本 没有考虑初始化策略
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta_base, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter
    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        y_spt = y_spt.to(torch.int64)
        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            # y_spt[i] =  y_spt[i].to(torch.int64)
            loss = F.cross_entropy(logits, y_spt[i])
            # target=y_spt[i]
            # input = F.softmax(logits, dim=1)
            # target=y_spt[i]
            # log_soft_out = torch.log(input)
            # # rms=
            # weight=torch.tensor([0.3,0.5,0.2],device='cuda:0')
            # loss_c = F.nll_loss(log_soft_out, target,weight=weight)
            #
            # loss_c = 0.0
            # for b in range(target.shape[0]):
            #     for i in range(target.shape[1]):
            #         for j in range(target.shape[2]):
            #             loss_c -= torch.log(input[b][target[b][i][j]][i][j])
            # # 求均值
            # c=loss_c / 9
            # print(loss / 9)



            grad = torch.autograd.grad(loss, self.net.parameters())
            # parameters=self.net.parameters()
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters()))) #更新参数
            #zip(grad, self.net.parameters()) 作用是把grad和self.net.parameters()组成一个元组
            # [(grad[0],self.net.parameters()[0]),(grad[1],self.net.parameters()[2]),...]
            # map 映射表示 某种映射关系

            # this is the loss and accuracy before first update
            y_qry = y_qry.to(torch.int64)
            with torch.no_grad(): #之后的内容不进行计算图构建 也就是不用跟踪反向梯度计算
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item() #若相同位置的两个元素相同，则返回True；若不同，返回False。
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.

                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct


            # # w1=rms[0]/(rms[0]+rms[1]+rms[2])

            # loss = F.cross_entropy(logits, y_spt[i],weight)



        # end of all tasks
        # sum over all losses on query set across all tasks
        cc=losses_q[-1]
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)

        return accs


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        y_spt = y_spt.to(torch.int64)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            y_qry = y_qry.to(torch.int64)
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        # print('pred_q', pred_q)
        # print('y_qry', y_qry)
        del net
        y_te = y_qry.cpu().numpy()
        predict = pred_q.cpu().numpy()
        acc = np.array(correct) / querysz
        # micro_precision = sklearn.metrics.precision_score(y_te, predict, labels=None, average='micro',
        #                                                   sample_weight=None)
        macro_precision = sklearn.metrics.precision_score(y_te, predict, labels=None, average='macro',
                                                          sample_weight=None)
        # micro_f1 = sklearn.metrics.f1_score(y_te, predict, labels=None, average='micro',
        #                                     sample_weight=None)
        macro_f1 = sklearn.metrics.f1_score(y_te, predict, labels=None, average='macro',
                                            sample_weight=None)
        # micro_recall = sklearn.metrics.recall_score(y_te, predict, labels=None, average='micro',
        #                                             sample_weight=None)
        macro_recall = sklearn.metrics.recall_score(y_te, predict, labels=None, average='macro',
                                                    sample_weight=None)
        accs = np.array(corrects) / querysz


        return accs,acc,macro_precision,macro_f1,macro_recall



def main():
    pass


if __name__ == '__main__':
    main()
