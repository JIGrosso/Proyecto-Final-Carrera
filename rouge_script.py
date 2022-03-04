# Script generico para utilizar rouge

import pandas as pd
import json
import re
import numpy as np
import difflib as dl

import spacy
import pytextrank
from spacy.lang.es import Spanish

def replace_semicolon(text, threshold=10):
    '''
    Get rid of semicolons.

    First split text into fragments between the semicolons. If the fragment
    is longer than the threshold, turn the semicolon into a period. O.w treat
    it as a comma.

    Returns new text
    '''
    new_text = ""
    for subset in re.split(';', text):
        subset = subset.strip()  # Clear off spaces
        # Check word count
        if len(subset.split()) > threshold:
            # Turn first char into uppercase
            new_text += ". " + subset[0].upper() + subset[1:]
        else:
            # Just append with a comma
            new_text += ", " + subset

    return new_text


def clean_text(text):
    # Reemplazar acentos
    text = A_TILDE_re.sub('a', text)
    text = E_TILDE_re.sub('e', text)
    text = I_TILDE_re.sub('i', text)
    text = O_TILDE_re.sub('o', text)
    text = U_TILDE_re.sub('u', text)

    # Documentos Legales Argentina
    text = text.replace('art.', 'articulo')

    # TODO Reescribir o eliminar
    # Get rid of enums as bullets or ` as bullets
    # text = BULLET_RE.sub(' ', text)

    # TODO Reescribir o eliminar
    # Clean html
    # text = text.replace('&lt;all&gt;', '')

    # TODO Evaluar
    # Remove annoying punctuation, that's not relevant
    # text = BAD_PUNCT_RE.sub('', text)

    # TODO Evaluar
    # removing newlines, tabs, and extra spaces.
    # text = WHITESPACE_RE.sub(' ', text)

    # TODO Evaluar
    # If we ended up with "empty" sentences - get rid of them.
    # text = EMPTY_SENT_RE.sub('.', text)

    # TODO Evaluar
    # Attempt to create sentences from bullets
    # text = replace_semicolon(text)

    # TODO Reescribir o eliminar
    # Fix weird period issues + start of text weirdness
    # text = re.sub('\.(?=[A-Z])', '  . ', text)
    # Get rid of anything thats not a word from the start of the text
    # text = FIX_START_RE.sub('', text)
    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent
    # text = FIX_PERIOD.sub(". \g<1>", text)

    # TODO Evaluar
    # Fix quotes
    # text = text.replace('``', '"')
    # text = text.replace('\'\'', '"')

    # TODO Reescribir o eliminar
    # Add special punct back in
    # text = text.replace('SECTION-HEADER', '<SECTION-HEADER>')

    return text


# LIMPIEZA DE TEXTO
def text_preprocessing(text_input):
    # TODO Verificar si no afecta al replace(art., articulo)
    # text_input = replace_semicolon(text_input, 10)

    text_input = clean_text(text_input)

    return text_input

from rouge import Rouge
from rouge import FilesRouge

#pip install rouge

if __name__ == "__main__":

     import argparse

     parser = argparse.ArgumentParser(description='Parser for PyTextRank parameters')
    #parser.add_argument('--filename', metavar='path', required=True, help='the name of the input file')

     args = parser.parse_args()

     nlp = spacy.load("es_core_news_lg")
     nlp_sentencizer = Spanish()

     nlp.add_pipe("textrank")
     nlp_sentencizer.add_pipe("sentencizer")

     # Definición de Expresiones Regulares
     # Esto permite que sean utilizadas mas tarde con otros métodos de RE.
     # fs. 227/233
     # Leyes 23.660
     A_TILDE_re = re.compile('[áÁ]')
     E_TILDE_re = re.compile('[éÉ]')
     I_TILDE_re = re.compile('[íÍ]')
     O_TILDE_re = re.compile('[óÓ]')
     U_TILDE_re = re.compile('[úÚ]')

     dataset = pd.read_json('./dataset/dataset-spa-test.json')

     targets = []
     data = {}

     # Auxiliares
     index = dataset.index
     lenght = len(index)
     print("Cantidad de documentos legales: " + str(lenght))

     # Sentencizer para el input
     for x in range(lenght):
          aux_line = dataset.at[x, 'lines']
          data['fallo ' + aux_line['bill_id']] = aux_line['text']
          data['target ' + aux_line['bill_id']] = aux_line['summary']
          # print('FALLO: ' + aux_line['bill_id'] + '\n' + aux_line['text'] + '\n')
          # print('SUMARIO: ' + aux_line['summary'] + '\n')
          # Sentencizer
          aux_doc = nlp_sentencizer(aux_line['summary'])
          targets.append(aux_doc)
          text = aux_line['text']

     nlp.max_length = 10 ** 7
     outputs = []

     for x in range(lenght):
          aux_sentences = ""  # Auxiliar para oraciones.
          aux_line = dataset.at[x, 'lines']  # Fallo Judicial.
          text_preprocessed = text_preprocessing(aux_line['text'])  # Limpieza del fallo.
          doc = nlp(text_preprocessed)  # Spacy process. Aqui se genera el sumario entre otras funcionalidades que ofrece el pipeline de Spacy.
          # print('Fallo: ' + aux_line['bill_id'] + '\n')
          # print('Sumario Generado: \n')
          # doc._.texrank.summary genera el sumario a partir de la info generada en 'doc'.
          # Basicamente summary tomas las frases que TextRank considera mas relevantes y las une en un solo objeto.
          for sentence in doc._.textrank.summary(limit_phrases=15, limit_sentences=5):
               aux_sentences = aux_sentences + str(sentence) + '\n'
               # TODO Tratar de obtener puntuación para oración
               # print(sentence)
          # print('\n')
          # TODO Verificar si este paso es necesario
          data['output ' + aux_line['bill_id']] = aux_sentences
          outputs.append(nlp_sentencizer(aux_sentences))  # Transformamos el sumario en oraciones.

     print (outputs)

     #rouge = Rouge()
     #files_rouge = FilesRouge()

     #dataset = pd.read_json('./rouge_data/generado.json')

     #scores = files_rouge.get_scores("rouge_data/generado.txt", "rouge_data/ideal.json")

     #print (scores)