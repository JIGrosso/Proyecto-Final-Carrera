import argparse
import json
import pandas as pd
import text_preprocessing
import spacy_summarizer
import rouge_script

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser for PyTextRank parameters')
    parser.add_argument('--filename', metavar='path', required=True, help='the name of the input file')

    args = parser.parse_args()

    print("Reading Text...")
    dataset = pd.read_json('./dataset/' + args.filename + '.json')

    print("Preprocessing Text...")
    preprocessed_text = text_preprocessing.process(dataset)

    print("Applying Summarizer...")
    spacy_summarizer.summary(preprocessed_text)

    print("Calculating ROUGE metrics...")
    rouge_script.get_rouge_scores()

    print("Process finished...")
