from rbmtf import RBM
import numpy as np
import rbm


def prepare_dataset(features):
    features_np = np.array(features)
    for element in features_np:
        print(element[0]) # Scores de la primer oración

    return features_np


# Input: Array de features
def improve_features(text_features):

    # Sum features
    global_scores = {}
    # Flag
    feature_position = 0
    for feature in text_features:
        sentence_position = 1
        for score in feature:
            # Initialize Dict
            if feature_position == 0:
                global_scores[str(sentence_position)] = 0
            global_scores[str(sentence_position)] = global_scores[str(sentence_position)] + score
            sentence_position += 1
        # Update Flag
        feature_position = 1
    # print(global_scores)

    # Sort Global Scores
    return sorted(global_scores.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)


def summary(text, features):
    prepare_dataset(features)
    n = 0
    for (sentence_position, score) in improve_features(features):
        if n < 5:
            print(int(sentence_position) - 1)
            print(text[1][int(sentence_position) - 1])
        n += 1
    return "Éxito"


def __test():
    test = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    # Visible layer: 4 nodes
    # Hidden layer: 3 nodes
    # Learning rate: 0,1 to 100
    rbm = RBM(4, 3, 0.1, 100)
    rbm.train(test)


__test()
