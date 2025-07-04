# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from keras.models import Model
from keras.layers import Input, Dense
from keras.activations import relu, sigmoid
 

# Basic ResNet model
def init_layer(L):
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            WeightNorm.apply(self.L, 'weight', dim=0)  # split the weight update component to direction and norm
        if outdim <= 200:
            self.scale_factor = 2  # a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10  # in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data / (L_norm + 0.00001)
        # matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * cos_dist
        return scores


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast,
                           self.bias.fast)  # weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        return out


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1])
        running_var = torch.ones(x.data.size()[1])
        if torch.cuda.is_available():
            running_mean = running_mean.cuda()
            running_var = running_var.cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)
        for layer in self.parametrized_layers:
            init_layer(layer)
        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


# Simple ResNet Block
class SimpleBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, outdim, kernel_size=3, stride=2 if half_res else 1, padding=1, bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1, bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]
        self.half_res = half_res
        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)
            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1, padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(indim, bottleneckdim, kernel_size=1, bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(bottleneckdim, bottleneckdim, kernel_size=3, stride=2 if half_res else 1, padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(bottleneckdim, outdim, kernel_size=1, bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU()
        self.parametrized_layers = [self.C1, self.BN1, self.C2, self.BN2, self.C3, self.BN3]
        self.half_res = half_res
        # if the input number of channels is not equal to the output, then need a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'
        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out
        out = self.relu(out)
        return out

'''
class ConvNet1(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        self.trunk = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()
        )
        self.final_feat_dim = 128
        # trunk = []
        # for i in range(depth):
        #     indim = 3 if i == 0 else 64
        #     outdim = 64
        #     B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
        #     trunk.append(B)
        # if flatten:
        #     trunk.append(Flatten())
        # self.trunk = nn.Sequential(*trunk)
        # self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out

'''


class ConvNet(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=128):
        super(ConvNet, self).__init__()

        self.fc1 = nn.Linear(1024, 256)  # input_dim 应该是1024
        self.fc21 = nn.Linear(256, 128)  # 上一层的输出是400，所以这里输入是400
        self.fc22 = nn.Linear(256, 128)  # 同上
        self.fc3 = nn.Linear(128, 256)  # latent_dim 应该是128，匹配上一层
        self.fc4 = nn.Linear(256, 1024)
        
    def forward(self, x):
        print("Input shape:", x.shape)
        x = F.relu(self.fc1(x))
        print("After fc1 shape:", x.shape)
        mu = self.fc21(x)
        logvar = self.fc22(x)
        print("mu shape:", mu.shape, "logvar shape:", logvar.shape)
        z = self.reparameterize(mu, logvar)
        print("z shape:", z.shape)
        x = self.decode(z)
        print("Output shape:", x.shape)
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 编码器
        self.fc1 = nn.Linear(1024, 256)
        self.fc21 = nn.Linear(256, 128)  # 均值
        self.fc22 = nn.Linear(256, 128)  # 对数方差

        # 解码器
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 1024)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        print('ahhhh')
        mu, logvar = self.encode(x)
        print(mu)
        print(logvar)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20531, 256),
            nn.ReLU(True)
        )
        # 定义解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(256, 20531),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 定义数据通过自编码器的方式
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        # 仅通过编码器获取编码后的表示
        return self.encoder(x)

    def decode(self, encoded_x):
        # 通过解码器重建数据
        return self.decoder(encoded_x)
 
 
 
 
 
 
 
 
 
 
 
 
    '''
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim),
            nn.ReLU(True)  # 潜在空间的表示
        )
        
        # 解码器定义
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 输出层使用Sigmoid激活函数
        )

        # trunk = []
        # for i in range(depth):
        #     indim = 3 if i == 0 else 64
        #     outdim = 64
        #     B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
        #     trunk.append(B)
        # if flatten:
        #     trunk.append(Flatten())
        # self.trunk = nn.Sequential(*trunk)
        # self.final_feat_dim = 1600

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
'''




class ConvNetNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetS(nn.Module):  # For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten=True):
        super(ConvNetS, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 4))  # only pooling for fist 4 layers
            trunk.append(B)
        if flatten:
            trunk.append(Flatten())
        # trunk.append(nn.BatchNorm1d(64))    #TODO remove
        # trunk.append(nn.ReLU(inplace=True)) #TODO remove
        # trunk.append(nn.Linear(64, 64))     #TODO remove
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        # out = torch.tanh(out) #TODO remove
        return out


class ConvNetSNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i in [0, 1]),
                          padding=0 if i in [0, 1] else 1)  # only first two layer has pooling and no padding
            trunk.append(B)
        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 5, 5]

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out


class ResNet(nn.Module):
    maml = False  # Default

    def __init__(self, block, list_of_num_layers, list_of_out_dims, flatten=True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        init_layer(conv1)
        init_layer(bn1)
        trunk = [conv1, bn1, relu, pool1]
        indim = 64
        for i in range(4):
            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]
        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]
        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


# Backbone for QMUL regression
class Conv3(nn.Module):
    def __init__(self):
        super(Conv3, self).__init__()
        self.layer1 = nn.Conv2d(3, 36, 3, stride=2, dilation=2)
        self.layer2 = nn.Conv2d(36, 36, 3, stride=2, dilation=2)
        self.layer3 = nn.Conv2d(36, 36, 3, stride=2, dilation=2)

    def return_clones(self):
        layer1_w = self.layer1.weight.data.clone().detach()
        layer2_w = self.layer2.weight.data.clone().detach()
        layer3_w = self.layer3.weight.data.clone().detach()
        return [layer1_w, layer2_w, layer3_w]

    def assign_clones(self, weights_list):
        self.layer1.weight.data.copy_(weights_list[0])
        self.layer2.weight.data.copy_(weights_list[1])
        self.layer3.weight.data.copy_(weights_list[2])

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = out.view(out.size(0), -1)
        return out


def Conv4():
    return ConvNet(4)

def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


def Conv4S():
    return ConvNetS(4)


def Conv4SNP():
    return ConvNetSNopool(4)


def ResNet10(flatten=True):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten)


def ResNet18(flatten=True):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten)


def ResNet34(flatten=True):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten)


def ResNet50(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 6, 3], [256, 512, 1024, 2048], flatten)


def ResNet101(flatten=True):
    return ResNet(BottleneckBlock, [3, 4, 23, 3], [256, 512, 1024, 2048], flatten)


# --- feature-wise transformation layer ---
class FeatureWiseTransformation2d_fw(nn.BatchNorm2d):
    feature_augment = False

    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(FeatureWiseTransformation2d_fw, self).__init__(num_features, momentum=momentum,
                                                             track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        if self.feature_augment:  # initialize {gamma, beta} with {0.3, 0.5}
            self.gamma = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.3)
            self.beta = torch.nn.Parameter(torch.ones(1, num_features, 1, 1) * 0.5)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros_like(x), torch.ones_like(x), weight, bias, training=True, momentum=1)

        # apply feature-wise transformation
        if self.feature_augment and self.training:
            gamma = (1 + torch.randn(1, self.num_features, 1, 1, dtype=self.gamma.dtype,
                                     device=self.gamma.device) * nn.functional.softplus(self.gamma,
                                                                                        beta=100)).expand_as(out)
            beta = (torch.randn(1, self.num_features, 1, 1, dtype=self.beta.dtype,
                                device=self.beta.device) * nn.functional.softplus(self.beta, beta=100)).expand_as(out)
            out = gamma * out + beta
        return out

# --- LSTMCell module for matchingnet ---
class LSTMCell(nn.Module):
    maml = False

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if self.maml:
            self.x2h = Linear_fw(input_size, 4 * hidden_size, bias=bias)
            self.h2h = Linear_fw(hidden_size, 4 * hidden_size, bias=bias)
        else:
            self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
            self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden=None):
        if hidden is None:
            hx = torch.zeors_like(x)
            cx = torch.zeros_like(x)
        else:
            hx, cx = hidden

        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_size, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, torch.tanh(cy))
        return (hy, cy)


# --- LSTM module for matchingnet ---
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=False, bidirectional=False):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.num_directions = 2 if bidirectional else 1
        assert (self.num_layers == 1)

        self.lstm = LSTMCell(input_size, hidden_size, self.bias)

    def forward(self, x, hidden=None):
        # swap axis if batch first
        if self.batch_first:
            x = x.permute(1, 0, 2)

        # hidden state
        if hidden is None:
            h0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
            c0 = torch.zeros(self.num_directions, x.size(1), self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h0, c0 = hidden

        # forward
        outs = []
        hn = h0[0]
        cn = c0[0]
        for seq in range(x.size(0)):
            hn, cn = self.lstm(x[seq], (hn, cn))
            outs.append(hn.unsqueeze(0))
        outs = torch.cat(outs, dim=0)

        # reverse foward
        if self.num_directions == 2:
            outs_reverse = []
            hn = h0[1]
            cn = c0[1]
            for seq in range(x.size(0)):
                seq = x.size(1) - 1 - seq
                hn, cn = self.lstm(x[seq], (hn, cn))
                outs_reverse.append(hn.unsqueeze(0))
            outs_reverse = torch.cat(outs_reverse, dim=0)
            outs = torch.cat([outs, outs_reverse], dim=2)

        # swap axis if batch first
        if self.batch_first:
            outs = outs.permute(1, 0, 2)
        return outs

# --- BatchNorm1d ---
class BatchNorm1d_fw(nn.BatchNorm1d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm1d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats)
        self.weight.fast = None
        self.bias.fast = None
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)

    def forward(self, x, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
        else:
            weight = self.weight
            bias = self.bias
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training,
                               momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device),
                               torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True,
                               momentum=1)
        return out

model_dict = dict(
    AE=AE,
    #vae=VariationalAutoencoder,
    VAE=VAE,
    Conv4=Conv4,
    Conv4S=Conv4S,
    Conv6=Conv6,
    ResNet10=ResNet10,
    ResNet18=ResNet18,
    ResNet34=ResNet34,
    ResNet50=ResNet50,
    ResNet101=ResNet101)
