import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np
import os
# from Residual import Residual


# zyg add 1125 --> XAI based attention module parameter
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#
#         self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # C x H x W -> 1 x H x W
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # C x H x W -> 1 x H x W
#         concat = torch.cat([avg_out, max_out], dim=1)  # 1 x H x W + 1 x H x W -> 2 x H x W
#         attention = self.sigmoid(self.conv1(concat))  # 2 x H x W -> 1 x H x W
#         return x * attention  # Apply attention to input


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)

    def forward(self, x, region_weights):
        # 生成空间注意力权重图
        attention_weights = torch.sigmoid(self.conv(x))

        # 假设region_weights是预先定义的，形状与attention_weights相同
        # 并且已经归一化到0-1之间
        if region_weights.eq(0).all():
            combined_weights = attention_weights 
        else:
            batch_size, channels, height, width = x.size()
            # 插值，使shap结果（region_weights）形状与x匹配
            region_weights = F.interpolate(region_weights, size=(height, width), mode='bilinear', align_corners=False)
            if region_weights == 0:
                combined_weights = attention_weights
            else:
                combined_weights = 2/3*attention_weights + region_weights / 3


        # 应用权重图到输入特征图
        out = x * combined_weights
        return out, combined_weights

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

        self.shap_weights = torch.zeros(50,1,2,2)  # shape 与attention_weights 相同

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

            # zyg add 1125
            elif name == 'spatial_attention':
                # self.attention = SpatialAttention(kernel_size=param[0] if param else 7)
                self.spatial_attention = SpatialAttention(param[0])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError


    def forward(self, x, vars = None, bn_training = True, region_weights = None):

        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weights.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        :region_weights应由XAI结果定义好，形状与最后一个卷积层的输出相匹配
         region_weights = ...  # XAI结果对应地图像区域权重图，形状为 (batch_size, 1, height, width)
        """
        if region_weights is None:
            region_weights = 0 
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
                #  zyg add 1125
                elif name == 'spatial_attention':
                    # 应用空间注意力模块
                    x, attention_weights = self.spatial_attention(x, self.shap_weights)


                elif name == 'flatten':
                    # print(x.shape)
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


                else:
                    raise NotImplementedError

        else:
            # vars = torch.load("net_vars.pth")
            vars_bn = torch.load("net_vars_bn.pth", weights_only=False, map_location='cuda:0')

            for name, param in self.config:
                # if name == 'lstm':
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
                #  zyg add 1125
                elif name == 'spatial_attention':
                    # 应用空间注意力模块
                    x, attention_weights = self.spatial_attention(x, region_weights)
                    # 注意：这里不需要更新idx，因为空间注意力模块没有可学习的参数

                elif name == 'flatten':
                    # print(x.shape)
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