import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class RBM():
    """
    nv and nh are the numbers of visible nodes and the number of hidden nodes.
    W is the weights for the visible nodes and hidden nodes.
    a is the bias for the probability of hidden nodes given visible node.
    b is the bias for the probability of visible nodes given hidden node.
    """
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)

    # The function takes argument x, which is the value of visible nodes. We use v to calculate the probability of hidden nodes
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # The function takes argument x, which is the value of hidden nodes. We use h to calculate the probability of visible nodes
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # v0 is the input vector containing the scores for all the features?.
    # vk is the visible nodes obtained after k samplings from visible nodes to hidden nodes.
    # ph0 is the vector of probabilities of hidden node equal to one at the first iteration given v0.
    # phk is the probabilities of hidden nodes given visible nodes vk at the kth iteration.
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
