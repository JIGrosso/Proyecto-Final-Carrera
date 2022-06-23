import json
import spacy
import pytextrank
from spacy.lang.es import Spanish


def summary(preprocessed_dataset):

    # Load Spanish tokenizer, tagger, parser and NER
    nlp = spacy.load("es_dep_news_trf")
    nlp_sentencizer = Spanish()

    # Add TextRank implementation to the pipeline || add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")
    nlp_sentencizer.add_pipe("sentencizer")

    # Auxiliares
    summaries = {}

    nlp.max_length = 10 ** 7  # TODO VERIFICAR PARA QUE SIRVE ESTO

    # Iterar sobre un Dict que viene desde el main
    # TODO El dict que viene deberían ser solo los inputs
    for text_id in preprocessed_dataset:
        aux_sentences = ""  # Auxiliar para oraciones.
        input_line = preprocessed_dataset[text_id]  # Leo el INPUT
        doc = nlp(input_line)  # Spacy process. Aqui se genera el sumario entre otras funcionalidades que ofrece el pipeline de Spacy.
        # doc._.texrank.summary genera el sumario a partir de la info generada en 'doc'.
        # Basicamente summary toma las frases que TextRank considera mas relevantes y las une en un solo objeto.
        for sentence in doc._.textrank.summary(limit_phrases=10, limit_sentences=4):
            aux_sentences = aux_sentences + str(sentence) + '\n'
            # TODO Tratar de obtener puntuación para oración
            # print(sentence)
        summaries[text_id] = aux_sentences
        # outputs.append(nlp_sentencizer(aux_sentences))  # Transformamos el sumario en oraciones.

    with open('./outputs/text_rank_summaries.json', 'w', encoding='utf8') as outfile:
        json.dump(summaries, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    print("Sumarios generados con éxito!")

    return True

