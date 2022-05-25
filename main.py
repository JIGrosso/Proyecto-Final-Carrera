import argparse
import json

import pandas as pd
import text_preprocessing
import text_features
import deep_learning_summarizer
import text_rank_summarizer
import rouge_script
import plots


def main():

    TextRank = False
    TextFeatures = False

    # Argumentos del main
    parser = argparse.ArgumentParser(description='Parser for PyTextRank parameters')
    parser.add_argument('--filename', metavar='path', required=True, help='the name of the input file')

    args = parser.parse_args()

    print("Reading Text...")
    dataset = pd.read_json('./dataset/' + args.filename + '.json')

    print("Preprocessing Text...")
    preprocessed_text, splitted_text = text_preprocessing.process(dataset)

    print("Applying Summarizer...")
    TextRank = text_rank_summarizer.summary(preprocessed_text)

    print("Getting Text Features...")
    features_vector = text_features.get_features_vector(splitted_text)

    print("Enhancing Text Features with Deep Learning...")
    TextFeatures = deep_learning_summarizer.generate_summaries(splitted_text, features_vector)

    print("Calculating ROUGE metrics...")
    if TextRank:
        rouge_scores_data = rouge_script.get_rouge_scores(target='targets', output='text_rank_summaries')

        print("Printing ROUGE metrics...")
        plots.print_rouge_recall(rouge_scores_data)
        plots.print_rouge_precision(rouge_scores_data)
        plots.print_rouge_f1_score(rouge_scores_data)

    if TextFeatures:
        rouge_scores_data = rouge_script.get_rouge_scores(target='targets', output='tf_dl_summaries')

        print("Printing ROUGE metrics...")
        plots.print_rouge_recall(rouge_scores_data)
        plots.print_rouge_precision(rouge_scores_data)
        plots.print_rouge_f1_score(rouge_scores_data)

    print("Process finished...")


if __name__ == "__main__":
    main()
