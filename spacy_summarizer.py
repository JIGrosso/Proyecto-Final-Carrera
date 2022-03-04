import pandas as pd
import json
import re
import numpy as np
import difflib as dl

import spacy
import pytextrank
from spacy.lang.es import Spanish


# SUMMARIZATION
def summary(preprocessed_dataset):

    # Load Spanish tokenizer, tagger, parser and NER
    nlp = spacy.load("es_core_news_lg")
    nlp_sentencizer = Spanish()

    # Add TextRank implementation to the pipeline
    nlp.add_pipe("textrank")
    nlp_sentencizer.add_pipe("sentencizer")

    # Auxiliares
    summaries = {}

    nlp.max_length = 10 ** 7  # TODO VERIFICAR PARA QUE SIRVE ESTO
    outputs = []

    # Iterar sobre un Dict que viene desde el main
    # TODO El dict que viene deberían ser solo los inputs
    for text_id in preprocessed_dataset:
        aux_sentences = ""  # Auxiliar para oraciones.
        input_line = preprocessed_dataset[text_id]  # Leo el INPUT
        doc = nlp(input_line)  # Spacy process. Aqui se genera el sumario entre otras funcionalidades que ofrece el pipeline de Spacy.
        # doc._.texrank.summary genera el sumario a partir de la info generada en 'doc'.
        # Basicamente summary toma las frases que TextRank considera mas relevantes y las une en un solo objeto.
        for sentence in doc._.textrank.summary(limit_phrases=15, limit_sentences=5):
            aux_sentences = aux_sentences + str(sentence) + '\n'
            # TODO Tratar de obtener puntuación para oración
            # print(sentence)
        # TODO Verificar si este paso es necesario
        summaries['output ' + text_id] = aux_sentences
        # outputs.append(nlp_sentencizer(aux_sentences))  # Transformamos el sumario en oraciones.

    print("Sumarios generados con éxito!")

    with open('./output_spacy_summarizer/summaries.json', 'w', encoding='utf8') as outfile:
        json.dump(summaries, outfile, indent=4, sort_keys=True, ensure_ascii=False)

