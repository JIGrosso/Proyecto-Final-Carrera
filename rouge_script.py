# Script generico para utilizar rouge

import pandas as pd
import json
from rouge import Rouge
from rouge import FilesRouge


def get_rouge_scores(target='targets', output='spacy_summaries'):

    rouge = Rouge()  # Para calcular Rouge desde Strings.
    files_rouge = FilesRouge()  # Para calcular Rouge desde Archivos.

    scores_response = {}

    with open('./outputs/' + target + '.json', encoding='utf8') as json_file:
        targets = json.load(json_file)
    with open('./outputs/' + output + '.json', encoding='utf8') as json_file:
        outputs = json.load(json_file)

    for text_id in outputs:
        scores = rouge.get_scores(targets[text_id], outputs[text_id])
        scores_response[text_id] = scores

    return scores_response


# get_rouge_scores()
