# Script generico para utilizar rouge

import pandas as pd
import json
from rouge import Rouge
from rouge import FilesRouge


def get_rouge_scores():

    rouge = Rouge()  # Para calcular Rouge desde Strings.
    files_rouge = FilesRouge()  # Para calcular Rouge desde Archivos.

    with open('./outputs/targets.json', encoding='utf8') as json_file:
        targets = json.load(json_file)
    with open('./outputs/spacy_summaries.json', encoding='utf8') as json_file:
        spacy_outputs = json.load(json_file)
    # targets = pd.read_json('./outputs/targets.json')
    # spacy_outputs = pd.read_json('./outputs/spacy_summaries.json')

    for text_id in spacy_outputs:
        scores = rouge.get_scores(targets[text_id], spacy_outputs[text_id])
        print(scores)


get_rouge_scores()
