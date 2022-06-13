import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


class RBM:
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
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    # The function takes argument x, which is the value of hidden nodes. We use h to calculate the probability of visible nodes
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    # v0 is the input vector containing the scores for all the features?.
    # vk is the visible nodes obtained after k samplings from visible nodes to hidden nodes.
    # ph0 is the vector of probabilities of hidden node equal to one at the first iteration given v0.
    # phk is the probabilities of hidden nodes given visible nodes vk at the kth iteration.
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


def train_rbm(training_set):
    training_set = torch.FloatTensor(training_set)

    # Configuración de la RBM
    nv = len(training_set[0])  # Nodos visibles
    nh = 5  # Nodos ocultos
    batch_size = 4
    rbm = RBM(nv, nh)  # Inicialización de la RBM

    nb_sentences = len(training_set)

    # Training the RBM model
    nb_epoch = 5  # Cantidad de épocas
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        for id_sentence in range(0, nb_sentences - batch_size, batch_size):
            vk = training_set[id_sentence:id_sentence + batch_size]
            v0 = training_set[id_sentence:id_sentence + batch_size]
            ph0, _ = rbm.sample_h(v0)
            for k in range(10):
                _, hk = rbm.sample_h(vk)
                _, vk = rbm.sample_v(hk)
                vk[v0 < 0] = v0[v0 < 0]
            phk, _ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
            s += 1.
        print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))


def enhance_scores(test_set_np):
    test_set = torch.FloatTensor(test_set_np)

    # Configuración de la RBM
    nv = len(test_set[0])  # Nodos visibles
    nh = 9  # Nodos ocultos
    batch_size = 4
    rbm = RBM(nv, nh)  # Inicialización de la RBM

    nb_sentences = len(test_set)

    test_loss = 0
    s = 0.
    for id_sentence in range(nb_sentences):
        v = test_set[id_sentence:id_sentence + 1]
        vt = test_set[id_sentence:id_sentence + 1]
        if len(vt[vt >= 0]) > 0:
            _, h = rbm.sample_h(v)
            _, v = rbm.sample_v(h)

            # Predicciones!
            tensor_to_array = v.numpy()
            # print(len(tensor_to_array[0]))
            print(s)

            test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
            s += 1.
    print('test loss: ' + str(test_loss / s))

