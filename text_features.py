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


def get_features_vector(sentence):

    # Thematic words
    get_thematic_words(sentence)


def __test():

    texto = 'apple mango apple orange orange apple guava mango mango'

    get_thematic_words(texto)


__test()


