import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import os
# from Residual import Residual



class Learner(nn.Module):
    """

    """

    def __init__(self, config, imgc, imgsz):
        """

        :param config: network config file, type:list of (string, list)
        :param imgc: 1 or 3
        :param imgsz:  28 or 84
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv1d':
                w = nn.Parameter(torch.ones(*param[:3]))
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            elif name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d': #反卷积
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
            elif name == 'residualcov':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
            # elif name is 'residualbn':
            #     # [ch_out]
            #     w = nn.Parameter(torch.ones(param[0]))
            #     self.vars.append(w)
            #     # [ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))
            #
            #     # must set requires_grad=False
            #     running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
            #     running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
            #     self.vars_bn.extend([running_mean, running_var])
            # elif name is 'residual':
            #     # [ch_out, ch_in, kernelsz, kernelsz]
            #     w = nn.Parameter(torch.ones(*param[:4]))
            #     # gain=1 according to cbfin's implementation
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     # [ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[0])))
            #     #BN
            #     # [ch_out]
            #     w = nn.Parameter(torch.ones(param[6]))
            #     self.vars.append(w)
            #     # [ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[6])))
            #     # must set requires_grad=False
            #     running_mean = nn.Parameter(torch.zeros(param[6]), requires_grad=False)
            #     running_var = nn.Parameter(torch.ones(param[6]), requires_grad=False)
            #     self.vars_bn.extend([running_mean, running_var])
            #
            #
            #     w = nn.Parameter(torch.ones(*param[7:11]))#8-13
            #     # gain=1 according to cbfin's implementation
            #     torch.nn.init.kaiming_normal_(w)
            #     self.vars.append(w)
            #     # [ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[7])))
            #     # BN
            #     # [ch_out]
            #     w = nn.Parameter(torch.ones(param[7]))
            #     self.vars.append(w)
            #     # [ch_out]
            #     self.vars.append(nn.Parameter(torch.zeros(param[13])))
            #     # must set requires_grad=False
            #     running_mean = nn.Parameter(torch.zeros(param[13]), requires_grad=False)
            #     running_var = nn.Parameter(torch.ones(param[13]), requires_grad=False)
            #     self.vars_bn.extend([running_mean, running_var])


                # w = nn.Parameter(torch.ones(*param[12:16]))
                # # gain=1 according to cbfin's implementation
                # torch.nn.init.kaiming_normal_(w)
                # self.vars.append(w)
                # # [ch_out]
                # self.vars.append(nn.Parameter(torch.zeros(param[12])))

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError






    # def extra_repr(self):
    #     info = ''
    #
    #     for name, param in self.config:
    #         if name is 'conv2d':
    #             tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
    #                   %(param[1], param[0], param[2], param[3], param[4], param[5],)
    #             info += tmp + '\n'
    #
    #         elif name is 'convt2d':
    #             tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
    #                   %(param[0], param[1], param[2], param[3], param[4], param[5],)
    #             info += tmp + '\n'
    #
    #         elif name is 'linear':
    #             tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
    #             info += tmp + '\n'
    #
    #         elif name is 'leakyrelu':
    #             tmp = 'leakyrelu:(slope:%f)'%(param[0])
    #             info += tmp + '\n'
    #
    #
    #         elif name is 'avg_pool2d':
    #             tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
    #             info += tmp + '\n'
    #         elif name is 'max_pool2d':
    #             tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
    #             info += tmp + '\n'
    #         elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
    #             tmp = name + ':' + str(tuple(param))
    #             info += tmp + '\n'
    #
    #         else:
    #             raise NotImplementedError
    #
    #     return info



    def forward(self, x, vars=None, bn_training=True):

        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weights.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars
        # print('self.vars',vars[0])
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        net_vars_save_path = f"{current_file_path}/net_vars.pth"
        net_vars_bn_save_path = f"{current_file_path}/net_vars_bn.pth"


        idx = 0
        bn_idx = 0
        flag=0
        X=0
        #cov->relu->
        #(8,1,28,28)->(8,64,13,13)->(8,64,13,13)->(8,64,13,13)->(8,64,7,7)->(8,64,3,3)
        if bn_training==True:
            torch.save(self.vars, net_vars_save_path)
            torch.save(self.vars_bn, net_vars_bn_save_path)
            for name, param in self.config:
                # if name is 'lstm':
                #     F.
                if name == 'conv1d':
                    w, b = vars[idx], vars[idx + 1] #64种1通道3乘3卷积核
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv1d(x, w, b, stride=param[3], padding=param[4])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)

                elif name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1] #64种1通道3乘3卷积核
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)
                elif name == 'convt2d':
                    w, b = vars[idx], vars[idx + 1]
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)
                elif name == 'linear':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.linear(x, w, b)
                    idx += 2
                    # print('forward:', idx, x.norm().item())
                elif name == 'bn':
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    idx += 2
                    bn_idx += 2

                elif name == 'flatten':
                    # print(x.shape)
                    # x = x.view(x.size(0), -1)
                    x = x.reshape(x.size(0), -1)
                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)
                elif name == 'relu':
                    if flag==2:
                        x = F.relu(x+X, inplace=param[0])
                        flag=0
                    else:
                        x = F.relu(x, inplace=param[0])
                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])
                elif name == 'residualcov':
                    if flag==0:
                        X = x
                    flag=flag+1
                    w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)



                    # w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # x= F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    # idx += 2
                    #
                    # w, b = vars[idx], vars[idx + 1]
                    # running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    # idx += 2
                    # bn_idx += 2
                    #
                    # x = F.relu(x, inplace=param[7])
                    #
                    # w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # x = F.conv2d(x, w, b, stride=param[12], padding=param[13])
                    # idx += 2
                    #
                    # w, b = vars[idx], vars[idx + 1]
                    # running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    # idx += 2
                    # bn_idx += 2
                    #
                    # X=X+x
                    # x = F.relu(X, inplace=param[15])

                else:
                    raise NotImplementedError

        else:
            # vars = torch.load("net_vars.pth")
            vars_bn = torch.load("net_vars_bn.pth", weights_only=False, map_location='cuda:0')

            for name, param in self.config:
                # if name is 'lstm':
                #     F.
                if name == 'conv1d':
                    w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv1d(x, w, b, stride=param[3], padding=param[4])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)

                elif name == 'conv2d':
                    w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)
                elif name == 'convt2d':
                    w, b = vars[idx], vars[idx + 1]
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)
                elif name == 'linear':
                    w, b = vars[idx], vars[idx + 1]
                    x = F.linear(x, w, b)
                    idx += 2
                    # print('forward:', idx, x.norm().item())
                elif name == 'bn':
                    w, b = vars[idx], vars[idx + 1]
                    running_mean, running_var = vars_bn[bn_idx], vars_bn[bn_idx + 1]
                    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=True)
                    idx += 2
                    bn_idx += 2

                elif name == 'flatten':
                    # print(x.shape)
                    # x = x.view(x.size(0), -1)
                    x = x.reshape(x.size(0), -1)
                elif name == 'reshape':
                    # [b, 8] => [b, 2, 2, 2]
                    x = x.view(x.size(0), *param)
                elif name == 'relu':
                    if flag == 2:
                        x = F.relu(x + X, inplace=param[0])
                        flag = 0
                    else:
                        x = F.relu(x, inplace=param[0])
                elif name == 'leakyrelu':
                    x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
                elif name == 'tanh':
                    x = F.tanh(x)
                elif name == 'sigmoid':
                    x = torch.sigmoid(x)
                elif name == 'upsample':
                    x = F.upsample_nearest(x, scale_factor=param[0])
                elif name == 'max_pool2d':
                    x = F.max_pool2d(x, param[0], param[1], param[2])
                elif name == 'avg_pool2d':
                    x = F.avg_pool2d(x, param[0], param[1], param[2])
                elif name == 'residualcov':
                    if flag == 0:
                        X = x
                    flag = flag + 1
                    w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # remember to keep synchrozied of forward_encoder and forward_decoder!
                    x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    idx += 2
                    # print(name, param, '\tout:', x.shape)

                    # w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # x= F.conv2d(x, w, b, stride=param[4], padding=param[5])
                    # idx += 2
                    #
                    # w, b = vars[idx], vars[idx + 1]
                    # running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    # idx += 2
                    # bn_idx += 2
                    #
                    # x = F.relu(x, inplace=param[7])
                    #
                    # w, b = vars[idx], vars[idx + 1]  # 64种1通道3乘3卷积核
                    # x = F.conv2d(x, w, b, stride=param[12], padding=param[13])
                    # idx += 2
                    #
                    # w, b = vars[idx], vars[idx + 1]
                    # running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
                    # x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                    # idx += 2
                    # bn_idx += 2
                    #
                    # X=X+x
                    # x = F.relu(X, inplace=param[15])

                else:
                    raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars