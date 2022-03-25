# Module to work with features for the Deep Learning technique

def get_thematic_words(input_text):

    # break the string into list of words
    text_list = input_text.split()

    # gives set of unique words
    unique_words = set(text_list)

    dictonary = {}

    for words in unique_words:
        dictonary[words] = text_list.count(words)

    sorted_tuples = sorted(dictonary.items(), key=lambda item: item[1], reverse=True)
    sorted_dict = {k: v for k, v in sorted_tuples}
    print(sorted_dict)


def sentence_position(sentence):

    return 1


def sentence_lenght(sentence):

    return 1


def sentence_to_paragraph(input_text):

    return 1


def proper_nouns(sentece):

    return 1


def get_features_vector(splitted_input_data):

    # input_data es un dict donde cada elemento es un array de parrafos
    for text_id in splitted_input_data:
        splitted_text = splitted_input_data[text_id]  # Array de parrafos
        # Dividir en oraciones
        for paragraph in splitted_text:
            get_thematic_words(paragraph)

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


