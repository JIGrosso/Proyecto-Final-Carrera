# Script generico para utilizar rouge

import pandas as pd
import json
from rouge import Rouge
from rouge import FilesRouge


def get_rouge_scores(target='targets', output='outputs'):
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
        scores = rouge.get_scores(outputs[text_id], targets[text_id])
        scores_response[text_id] = scores

    return scores_response

