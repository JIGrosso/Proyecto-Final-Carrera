# Module to work with features for the Deep Learning technique
import math
import nltk
from collections import Counter


def get_thematic_words(input_text):
    word_list = []
    for sentence in input_text:
        # break the string into list of words
        for w in sentence.split():
            word_list.append(w)

    counts = Counter(word_list)
    number_of_words = len(counts)

    # Most common words
    most_common = counts.most_common(10)
    thematic_words = []
    for word in most_common:
        thematic_words.append(str(word[0]))
    print(thematic_words)
    # Scores
    scores = []
    for sentence in input_text:
        score = 0
        for word in sentence.split():
            if word in thematic_words:
                score = score + 1
        score = 1.0 * score / number_of_words
        scores.append(score)

    print(scores)
    return scores


def sentence_position(input_text):
    # Aux
    N = len(input_text)
    th = 0.2 * N
    min_var = th * N
    max_var = th * 2 * N

    scores = []
    position = 1
    last_sentence_position = N

    for sentence in input_text:
        score = 0
        if position == 1 or position == last_sentence_position:
            score = 1
        else:
            score = math.cos((position - min_var) * ((1 / max_var) - min_var))
        scores.append(score)
        position += 1
    print(scores)
    return scores


def sentence_lenght(input_text):
    scores = []
    for sentence in input_text:
        score = 0
        length = len(sentence.split())
        if length > 3:
            score = length
        scores.append(score)

    print(scores)

    return scores


def sentence_to_paragraph(input_text):

    return 1


def proper_nouns(input_text):
    tags = []
    for sentence in input_text:
        tag = nltk.pos_tag(sentence.split())
        tags.append(tag)
    # TODO Continuar revisando los tags y calcular scores
    print(tags)


def get_features_vector(splitted_input_data):

    # input_data es un dict donde cada elemento es un array de parrafos
    for text_id in splitted_input_data:
        splitted_text = splitted_input_data[text_id]  # Array de parrafos
        # Dividir en oraciones
        get_thematic_words(splitted_input_data[text_id])
        sentence_position(splitted_input_data[text_id])
        sentence_lenght(splitted_input_data[text_id])
        proper_nouns(splitted_input_data[text_id])

    # Thematic words
    # get_thematic_words(sentence)

    # Calcular Posición de las oraciones
    # Calcular Longitud de las oraciones
    # Calcular Posición de las oraciones respecto al párrafo que pertenecen
    # Calcular Sustantivos Propios
    # Calcular Mamed Entities
    # Calcular TF ISF
    # Calcular Sentence Similarity


def __test():

    texto = 'apple mango apple orange orange apple guava mango mango'

    get_thematic_words(texto)


# __test()


