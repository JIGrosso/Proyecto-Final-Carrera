import pandas as pd
import json
import spacy
from nltk.tokenize import sent_tokenize
from spacy.lang.es import Spanish


def analyze_sentences_nltk(filename):

    dataset = pd.read_json('./dataset/' + filename + '.json')

    output_data = {}  # Dict con todos los targets
    sentence_length_data = {}
    index = dataset.index  # Longitud del dataset
    lenght = len(index)  # Longitud del dataset
    print("Cantidad de documentos legales: " + str(lenght))

    # Itero sobre el Dataset y lo fragmento
    for x in range(lenght):
        json_line = dataset.at[x, 'lines']
        output_data[json_line['bill_id']] = json_line['summary']

        sentence_length_data[json_line['bill_id']] = len(sent_tokenize(json_line['summary']))

    # Guardado
    with open(f'./outputs/sentence_analyzer_{filename}_nltk.json', 'w', encoding='utf8') as outfile:
        json.dump(sentence_length_data, outfile, indent=4, sort_keys=False, ensure_ascii=False)


def analyze_sentences_spacy(filename):
    nlp_sentencizer = Spanish()
    nlp_sentencizer.add_pipe("sentencizer")

    dataset = pd.read_json('./dataset/' + filename + '.json')

    output_data = {}  # Dict con todos los targets
    sentence_length_data = {}
    index = dataset.index  # Longitud del dataset
    lenght = len(index)  # Longitud del dataset
    print("Cantidad de documentos legales: " + str(lenght))

    # Itero sobre el Dataset y lo fragmento
    for x in range(lenght):
        json_line = dataset.at[x, 'lines']

        doc = nlp_sentencizer(json_line['summary'])
        sentence_count = 0
        for sent in doc.sents:
            sentence_count += 1

        sentence_length_data[json_line['bill_id']] = sentence_count

    # Guardado
    with open(f'./outputs/sentence_analyzer_{filename}_spacy.json', 'w', encoding='utf8') as outfile:
        json.dump(sentence_length_data, outfile, indent=4, sort_keys=False, ensure_ascii=False)


def get_largest_summary_nltk(filename):
    dataset = pd.read_json(f'./outputs/sentence_analyzer_{filename}_nltk.json', typ='series')

    text_id = 0
    counts = 0

    for key, value in dataset.items():
        if value > counts:
            counts = value
            text_id = key

    print(f"[NLTK] Sumario con mayor cantidad de oraciones: {text_id} - {counts}")


def get_largest_summary_spacy(filename):
    dataset = pd.read_json(f'./outputs/sentence_analyzer_{filename}_spacy.json', typ='series')

    text_id = 0
    counts = 0

    for key, value in dataset.items():
        if value > counts:
            counts = value
            text_id = key

    print(f"[SPACY] Sumario con mayor cantidad de oraciones: {text_id} - {counts}")


def main():
    file = 'input_analyzer_1234937'
    analyze_sentences_nltk(file)
    analyze_sentences_spacy(file)
    get_largest_summary_nltk(file)
    get_largest_summary_spacy(file)


if __name__ == "__main__":
    main()
