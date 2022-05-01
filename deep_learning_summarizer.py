from rbmtf import RBM
import numpy as np


# Input: Array de features
def improve_features(text_features):
    for feature in text_features:
        print(feature)


def __test():
    test = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    # Visible layer: 4 nodes
    # Hidden layer: 3 nodes
    # Learning rate: 0,1 to 100
    rbm = RBM(4, 3, 0.1, 100)
    temp = rbm.train(test)
    print(temp)
