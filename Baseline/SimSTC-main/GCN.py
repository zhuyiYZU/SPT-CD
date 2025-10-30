import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

'''
标准的图卷积层。该层可以通过邻接矩阵（adj）对输入特征（inputs）进行卷积操作，从而学习节点的嵌入表示。
'''

class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            torch.empty([in_features, out_features], dtype=torch.float),
            requires_grad=True)
        if bias:
            self.bias = Parameter(torch.empty([out_features], dtype=torch.float))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()  #用于给权重和偏置赋初始值

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))  #通过标准差计算权重初始化的范围。标准差依据输入特征的数量（in_features）来设定
        self.weight.data.uniform_(-stdv, stdv)  #将权重参数初始化为均匀分布的值，范围在 [-stdv, stdv] 之间。
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj, inputs, identity=False): 
        if identity:
            return torch.sparse.mm(adj, self.weight)
        return torch.sparse.mm(adj, torch.sparse.mm(inputs, self.weight))
    