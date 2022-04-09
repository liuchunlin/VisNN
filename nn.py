import random

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter


class Link:
    def __init__(self, srcnode, dstnode, weight):
        self.srcnode: Node = srcnode
        self.dstnode: Node = dstnode
        self.weight = Parameter(torch.zeros(1, dtype=torch.float32))
        self.weight.data.fill_(weight)
        self.graph = None


class Node(nn.Module):
    def __init__(self, activefunc=None):
        super(Node, self).__init__()
        self.outlinks = []
        self.inlinks = []
        self.params = Parameter(torch.rand(7, dtype=torch.float32))
        self.params.data[0].fill_(0)
        self.xy = None
        self.output = None
        self.activefunc = self.myactivefunc
        if activefunc != None:
            self.activefunc = activefunc

    def connect(self, other, weight):
        link = Link(self, other, weight)
        self.outlinks.append(link)
        other.inlinks.append(link)
        linkname = f"link-{self.name}-{other.name}"
        self.register_parameter(f"{linkname}-weight", link.weight)

    def forward(self):
        input = 0
        for link in self.inlinks:
            input = input + link.srcnode.output * link.weight
        input = input + self.params[0]
        return self.activefunc(input)

    def myactivefunc(self, input):
        #return torch.relu(input)
        #return torch.sigmoid(input)
        #return torch.tanh(input)

        a = self.params[1:]
        return a[0] * torch.tanh(input) * torch.sin(a[1]*input + a[2]) + a[3]*input + a[4]
        #return torch.tanh(input) * torch.sin(a[1]*input + a[2]) + a[3]*input


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.layers = []

    def addlayer(self, nodes, autolink=True):
        if isinstance(nodes, int):
            nodes = [Node() for i in range(nodes)]
        elif not isinstance(nodes, list):
            raise "param nodes must be int or list"

        self.layers.append(nodes)
        for i in range(len(nodes)):
            nodename = f"node-{len(self.layers)-1}-{i}"
            nodes[i].name = nodename
            self.add_module(nodename, nodes[i])

        if autolink and len(self.layers) > 1:
            for inode in self.layers[-2]:
                for jnode in self.layers[-1]:
                    inode.connect(jnode, random.random())

    def forward(self, data):
        inputlayer = self.layers[0]
        for i in range(len(inputlayer)):
            inputlayer[i].output = data[:, i]

        for layer in self.layers[1:]:
            for node in layer:
                node.output = node()

        output = []
        for node in self.layers[-1]:
            output.append(node.output[:, np.newaxis])
        output = torch.hstack(output)
        return torch.softmax(output, dim=1, dtype=torch.float32)


class TorchNetwork(nn.Module):
    def __init__(self):
        super(TorchNetwork, self).__init__()
        self.layer_seq = nn.Sequential(
            nn.Linear(2, 800),
            nn.ReLU(),
            nn.Linear(800, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layer_seq(x)
