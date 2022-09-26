# Script generico para utilizar rouge

import pandas as pd
import json
from rouge import Rouge


def get_rouge_scores(target='targets', output='outputs', persist=False):
    """
    target: Nombre del archivo que contiene los targets
    output: Nombre del archivo que contiene los outputs
    """

    rouge = Rouge()  # Para calcular Rouge desde Strings.

    scores_response = {}

    with open('./outputs/' + target + '.json', encoding='utf8') as json_file:
        targets = json.load(json_file)
    with open('./outputs/' + output + '.json', encoding='utf8') as json_file:
        outputs = json.load(json_file)

    for text_id in outputs:
        # rouge.get_scores(hypothesis, reference)
        # Unigram recall reflects the proportion of words in X (reference summary sentence) that are also present in Y (candidate summary sentence)
        # Unigram precision is the proportion of words in Y that are also in X
        scores = rouge.get_scores(outputs[text_id], targets[text_id])
        scores_response[text_id] = scores

    if persist:
        # Persistir scores
        dump_path = './outputs/' + output + '_rouge_scores.json'
        with open(dump_path, 'w', encoding='utf8') as outfile:
            json.dump(scores_response, outfile, indent=4, sort_keys=False, ensure_ascii=False)

    return scores_response

