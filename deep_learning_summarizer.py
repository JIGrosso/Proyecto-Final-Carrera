from rbmtf import RBM
import numpy as np


def __sort_features():
    return 1


# Input: Array de features
def improve_features(text_features):
    global_scores = {}
    #  Flag
    feature_position = 0
    for feature in text_features:
        sentence_position = 1
        for score in feature:
            # Initialize Dict
            if feature_position == 0:
                global_scores["Sentence " + str(sentence_position)] = 0
            global_scores["Sentence " + str(sentence_position)] = global_scores["Sentence " + str(sentence_position)] + score
            sentence_position += 1
        # Update Flag
        feature_position = 1
    print(global_scores)


def summary():

    return "Sumario generado:"


def __test():
    test = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    # Visible layer: 4 nodes
    # Hidden layer: 3 nodes
    # Learning rate: 0,1 to 100
    rbm = RBM(4, 3, 0.1, 100)
    temp = rbm.train(test)
    print(temp)
