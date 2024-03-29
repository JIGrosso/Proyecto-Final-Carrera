# Module to work with features for the Deep Learning technique
import collections
import math
import nltk
from collections import Counter
import re
from nltk.tokenize import sent_tokenize


def __tokenize_sentence(input_sentence):
    BAD_PUNCT_RE = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}():.,~_'), re.UNICODE)
    return BAD_PUNCT_RE.sub('', input_sentence).split()


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
    # Scores
    scores = []
    for sentence in input_text:
        score = 0
        for word in sentence.split():
            if word in thematic_words:
                score = score + 1
        score = 1.0 * score / number_of_words
        scores.append(score)

    # print(scores)
    # print('get_thematic_words len(scores): ' + str(len(scores)))
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
    # print(scores)
    # print('sentence_position len(scores): ' + str(len(scores)))
    return scores


def sentence_length(input_text):
    scores = []
    word_list = []

    for sentence in input_text:
        for w in sentence.split():
            word_list.append(w)

    counts = Counter(word_list)
    number_of_words = len(counts)

    for sentence in input_text:
        score = 0
        length = len(sentence.split())
        if length > 3:
            score = length/number_of_words
        scores.append(score)

    # print(scores)
    # print('sentence_length len(scores): ' + str(len(scores)))
    return scores


def sentence_to_paragraph(input_p):

    scores = []

    for paragraph in input_p:
        for sentence in paragraph:
            if sentence == paragraph[0]:
                scores.append(1)
            else:
                scores.append(0)

    # print(scores)
    # print('sentence_to_paragraph len(scores): ' + str(len(scores)))
    return scores


def __tagger(sentence):
    return nltk.pos_tag(sentence.split())


def proper_nouns(input_text):
    scores = []
    for sentence in input_text:
        score = 0
        length = len(sentence.split())
        tag_list = __tagger(sentence)
        for tag in tag_list:
            if tag[1] == "NNP" or tag[1] == "NNPS":
                score += 1
        scores.append(score/float(length))

    # print(scores)
    # print('proper_nouns len(scores): ' + str(len(scores)))
    return scores


def __is_number(word):
    try:
        float(word)
        return True
    except ValueError:
        return False


def numerals(input_text):
    scores = []
    for sentence in input_text:
        sentence_split = sentence.split()
        score = 0
        for word in sentence_split:
            if __is_number(word):
                score += 1
        scores.append(score/float(len(sentence_split)))

    # print(scores)
    # print('numerals len(scores): ' + str(len(scores)))
    return scores


def __extract_named_entities(c):
    entity_names = []
    if hasattr(c, 'label') and c.label:
        if c.label() == 'NE':
            entity_names.append(' '.join(child[0] for child in c))
        else:
            for child in c:
                entity_names.extend(__extract_named_entities(child))

    return entity_names


def named_entities(input_text):
    # TODO Continuar con este feature. Las NE que reconoce son erroneas. Creo que puede ser debido al idioma. Probar con el NE Recognition de Spacy.
    scores = []
    for sentence in input_text:
        length = len(sentence.split())
        tagged_sentence = __tagger(sentence)
        chunked_sentence = nltk.ne_chunk(tagged_sentence, binary=True)
        entity_names = []
        for c in chunked_sentence:
            entity_names.extend(__extract_named_entities(c))
        set(entity_names)
        scores.append(len(entity_names)/float(length))

    # print(scores)
    # print('named_entities len(scores): ' + str(len(scores)))
    return scores


def tf_isf(input_text):
    # TODO Eliminar palabras de menos de tres caracteres
    scores = []
    for sentence in input_text:
        sum_tfisf = 0
        sentence_len = len(sentence.split())

        counts = collections.Counter(__tokenize_sentence(sentence.lower()))
        for word in counts.keys():
            count_word = 0
            for aux_sentence in input_text:
                aux_counts = collections.Counter(__tokenize_sentence(aux_sentence.lower()))
                if word in aux_counts.keys():
                    count_word = count_word + aux_counts[word]
            sum_tfisf += counts[word] * count_word
        try:
            scores.append(math.log(sum_tfisf/sentence_len))
        except:
            scores.append(0)

    # print(scores)
    # print('tf_isf len(scores): ' + str(len(scores)))
    return scores


def __get_centroid_sentence(scores_tf_isf):
    value = 0
    position = 0
    i = 0
    for score in scores_tf_isf:
        if score > value:
            value = score
            position = i
        i += 1

    return position


def centroid_similarity(input_text, scores_tf_isf):
    scores = []
    position = __get_centroid_sentence(scores_tf_isf)
    centroid_sentence = input_text[position]
    tokenized_centroid_sentence = __tokenize_sentence(centroid_sentence.lower())

    for sentence in input_text:
        # tokenization
        tokenized_sentence = __tokenize_sentence(sentence.lower())

        l1 = []
        l2 = []

        # form a set containing keywords of both strings
        rvector = {}
        rvector = set(tokenized_centroid_sentence).union(set(tokenized_sentence))

        for w in rvector:
            if w in set(tokenized_centroid_sentence):
                l1.append(1)  # create a vector
            else:
                l1.append(0)
            if w in set(tokenized_sentence):
                l2.append(1)
            else:
                l2.append(0)
        c = 0

        # cosine formula
        for i in range(len(rvector)):
            c += l1[i] * l2[i]
        try:
            cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
            scores.append(cosine)
        except:
            scores.append(0)

    # print(scores)
    # print('centroid_similarity len(scores): ' + str(len(scores)))
    return scores


def get_features_vector(splitted_input_data):

    features_vector = {}

    # splitted_input_data es un dict donde cada elemento es un texto del cual se extraen features
    for text_id in splitted_input_data:
        text_scores = []
        splitted_text = splitted_input_data[text_id]  # Array de parrafos de dos dimensiones

        '''
        splitted_text[0] -> array de array. Contiene array de parrafos, cada parrafo es un array de oraciones
        splitted_text[1] -> array de oraciones, sin especificar por parrafo
        splitted_text[2] -> array de oraciones, sin especificar por parrafo, con stop-words incluidas
        '''

        text_scores.append(get_thematic_words(splitted_text[1]))
        text_scores.append(sentence_position(splitted_text[1]))
        text_scores.append(sentence_length(splitted_text[1]))
        text_scores.append(sentence_to_paragraph(splitted_text[0]))
        text_scores.append(proper_nouns(splitted_text[1]))
        text_scores.append(numerals(splitted_text[1]))
        text_scores.append(named_entities(splitted_text[1]))
        scores_tf_isf = tf_isf(splitted_text[1])
        text_scores.append(scores_tf_isf)
        text_scores.append(centroid_similarity(splitted_text[1], scores_tf_isf))

        '''
        text_scores[0] -> Thematic words 
        text_scores[1] -> Sentence position
        text_scores[2] -> Sentence length
        text_scores[3] -> Sentence to paragraph
        text_scores[4] -> Proper noun
        text_scores[5] -> Numerals
        text_scores[6] -> Named entities
        text_scores[7] -> TF-ISF
        text_scores[8] -> Centroid simlarity
        '''

        features_vector[text_id] = text_scores

    return features_vector


def __test():

    texto = 'apple mango apple orange orange apple guava mango mango'

    get_thematic_words(texto)


# __test()


