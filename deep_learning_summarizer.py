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
    # rbm.train_rbm(prepare_dataset(text_features))
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


def sort_features(text_features):

    enhanced_features = prepare_dataset(text_features)

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

    # True -> Uses the RBM. False -> Does not use the RBM
    rbm_process = True

    text_summary = ''
    i = 0
    n = 0

    # Used to compare results with or without the use of the RBM
    if rbm_process:
        improved_features = improve_features(features)
    else:
        improved_features = sort_features(features)

    jaccard_similarities = {}

    sentences = []

    best_sentence_position = int(improved_features[i][0]) - 1
    best_sentence_text = text[2][best_sentence_position]

    sentences.append((best_sentence_text, best_sentence_position))

    i += 1
    while i < len(improved_features)/2:
        sent_position = int(improved_features[i][0]) - 1
        # text[1][0] = Texto sin stop words - Oracion con mas score
        jaccard_similarities[sent_position] = _jaccard_similarity(text[1][best_sentence_position], text[1][sent_position])
        i += 1

    sorted_similarities = sorted(jaccard_similarities.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    for (sentence_position, score) in sorted_similarities:
        if n < (15-1):
            sentences.append((text[2][int(sentence_position) - 1], sentence_position))
        n += 1

    sorted_sentences = sorted(sentences, key=lambda kv: kv[1], reverse=False)

    text_summary = ''
    for (text, position) in sorted_sentences:
        text_summary = text_summary + text + " "

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
