import argparse
import json
import pandas as pd
import text_preprocessing
import spacy_summarizer

if __name__ == "__main__":

    print("Reading Text...")
    print("Preprocessing Text...")
    print("Applying Summarizer...")
    print("Process finished...")

    parser = argparse.ArgumentParser(description='Parser for PyTextRank parameters')
    parser.add_argument('--filename', metavar='path', required=True, help='the name of the input file')

    args = parser.parse_args()

    dataset = pd.read_json('./dataset/' + args.filename + '.json')

    preprocessed_text = text_preprocessing.process(dataset)

    with open('./outputs/preprocessed_text.json', 'w', encoding='utf8') as outfile:
        json.dump(preprocessed_text, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    # nlp.max_length = 10 ** 7 #TODO VERIFICAR PARA QUE SIRVE ESTO

