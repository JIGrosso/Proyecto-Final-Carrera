import rbm
import json

import numpy as np


def prepare_dataset(features):
    features_np = np.array(features)
    """
        Peliculas -> Features
        Usuarios -> Oraciones
        Puntuaciones -> Scores
        
        Size features_np: 9 x CantidadOraciones
        
        Columnas -> Features
        Filas -> Oraciones
        Celdas -> Scores
        
    """

    test_set = np.transpose(features_np)

    return test_set


def _jaccard_similarity(sentence_a, sentence_b):
    words_a = set(sentence_a.split())
    words_b = set(sentence_b.split())
    # print(words_a)
    # print(words_b)

    intersection = len(words_a.intersection(words_b))
    # print(intersection)
    union = len(words_a) + len(words_b)
    # print(union)
    return float(intersection/union)


# Input: Array de features
def improve_features(text_features):

    # Llamada a método de la rbm
    rbm.train_rbm(prepare_dataset(text_features))
    enhanced_features = rbm.enhance_scores(prepare_dataset(text_features))

    # Calculate global scores
    global_scores = {}

    # Flag
    sentence_position = 1
    for sentence in enhanced_features:
        feature_position = 0
        for feature_score in sentence:
            # Initialize Dict
            if feature_position == 0:
                global_scores[str(sentence_position)] = feature_score
            else:
                global_scores[str(sentence_position)] = global_scores[str(sentence_position)] + feature_score
            feature_position += 1
        # Update Flag
        sentence_position += 1
    # print(global_scores)

    # Sort Global Scores
    return sorted(global_scores.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)


def summary(text, features):

    text_summary = ''
    i = 0
    n = 0

    improved_features = improve_features(features)
    jaccard_similarities = {}

    while i < len(improved_features)/2:

        if i == 0:
            i += 1
            continue
        else:
            sent_position = int(improved_features[i][0]) - 1
            # text[1][0] = Texto sin stop words - Oracion con mas score
            jaccard_similarities[sent_position] = _jaccard_similarity(text[1][0], text[1][sent_position])
            i += 1

    print(sorted(jaccard_similarities.items(), key=lambda kv:(kv[1], kv[0]), reverse=True))

    for (sentence_position, score) in improved_features:
        if n < 5:
            # print(int(sentence_position) - 1)
            sentence = text[2][int(sentence_position) - 1]
            # print(sentence)
            text_summary = text_summary + sentence + "\n"
        n += 1

    return text_summary


def generate_summaries(dataset, features_vector):

    summaries = {}

    for text_id in dataset:
        summaries[text_id] = summary(dataset[text_id], features_vector[text_id])

    with open('./outputs/tf_dl_summaries.json', 'w', encoding='utf8') as outfile:
        json.dump(summaries, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    print("Sumarios generados con éxito!")

    return True

# def __test():
    # test = np.array([[0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1]])
    # Visible layer: 4 nodes
    # Hidden layer: 3 nodes
    # Learning rate: 0,1 to 100
    # rbm = RBM(4, 3, 0.1, 100)
    # rbm.train(test)


# __test()
