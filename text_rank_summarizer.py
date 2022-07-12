import json
import spacy
import pytextrank
from spacy.lang.es import Spanish


def summary(preprocessed_dataset):

    # Load Spanish tokenizer, tagger, parser and NER
    #nlp = spacy.load("es_dep_news_trf")
    nlp = spacy.load("es_core_news_sm")

    # Add TextRank implementation to the pipeline || add PyTextRank to the spaCy pipeline
    nlp.add_pipe("textrank")

    # Auxiliares
    summaries = {}

    nlp.max_length = 10 ** 7  # Límite de longitud del texto a procesar

    # Iterar sobre un Dict que viene desde el main
    for text_id in preprocessed_dataset:
        aux_sentences = ""  # Auxiliar para oraciones.
        input_line = preprocessed_dataset[text_id]  # Leo el INPUT
        doc = nlp(input_line)  # Spacy process. Aqui se genera el sumario entre otras funcionalidades que ofrece el pipeline de Spacy.
        # doc._.texrank.summary genera el sumario a partir de la info generada en 'doc'.
        # Basicamente summary toma las frases que TextRank considera mas relevantes y las une en un solo objeto.
        i = 1
        for sentence in doc._.textrank.summary(limit_phrases=5, limit_sentences=2):
            aux_sentences = aux_sentences + str(sentence) + '\n'
            # TODO Obtener puntuación para oración
            print(i)
            i += 1
            print(sentence)
        summaries[text_id] = aux_sentences

    with open('./outputs/text_rank_summaries.json', 'w', encoding='utf8') as outfile:
        json.dump(summaries, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    print("Sumarios generados con éxito!")

    return True

