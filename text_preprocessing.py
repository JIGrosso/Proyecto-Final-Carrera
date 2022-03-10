import pandas as pd
import json
import re
import numpy as np
import difflib as dl

import spacy
import pytextrank
from spacy.lang.es import Spanish

# Definición de Expresiones Regulares
# Esto permite que sean utilizadas mas tarde con otros métodos de RE.
# fs. 227/233
# Excma.
# -v. cláusula séptima-

# TODO Limpiar
A_TILDE_re = re.compile('[áÁ]')
E_TILDE_re = re.compile('[éÉ]')
I_TILDE_re = re.compile('[íÍ]')
O_TILDE_re = re.compile('[óÓ]')
U_TILDE_re = re.compile('[úÚ]')
DOT_BETWEEN_NUMBERS_re = re.compile(r"\b[0-9]{1,2}(?:\.[0-9]{3})+\b")


def __replace_semicolon(text, threshold=10):
    """
    Get rid of semicolons.

    First split text into fragments between the semicolons. If the fragment
    is longer than the threshold, turn the semicolon into a period. O.w treat
    it as a comma.

    Returns new text
    """
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


def __clean_text(text):
    refined_text = text
    # TODO Limpiar
    # BAD_PUNCT_RE = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
    # BULLET_RE = re.compile('\n[\ \t]*`*\([a-zA-Z0-9]*\)')
    # DASH_RE = re.compile('--+')
    # WHITESPACE_RE = re.compile('\s+')
    # EMPTY_SENT_RE = re.compile('[,\.]\ *[\.,]')
    # FIX_START_RE = re.compile('^[^A-Za-z]*')
    # FIX_PERIOD = re.compile('\.([A-Za-z])')

    # Documentos Legales Argentina
    refined_text = refined_text.replace('art.', 'artículo')
    refined_text = refined_text.replace('arts.', 'artículos')
    refined_text = refined_text.replace('Dr.', 'doctor')
    refined_text = refined_text.replace('Dra.', 'doctora')
    refined_text = refined_text.replace('Dres.', 'doctores')
    refined_text = refined_text.replace('Sr.', 'señor')
    refined_text = refined_text.replace('Sra.', 'señora')
    refined_text = refined_text.replace('Sres.', 'señores')
    refined_text = refined_text.replace('pag.', 'página')
    refined_text = refined_text.replace('inc.', 'inciso')
    refined_text = refined_text.replace('\r\r\n', '')

    # Elimina los puntos entre números. Ejemplo : 16.233 -> 16233
    refined_text = re.sub(DOT_BETWEEN_NUMBERS_re, lambda x: x.group().replace(".", ""), refined_text)

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

    return refined_text


def process(dataset):

    # Me sirve para ver la estructura del dataset - returns Panda's object
    # dataset = pd.read_json('./dataset/'+args.filename+'.json')

    # Auxiliares
    input_data = {}  # Dict con todos los inputs
    target_data = {}  # Dict con todos los targets
    index = dataset.index  # Longitud del dataset
    lenght = len(index)  # Longitud del dataset
    print("Cantidad de documentos legales: " + str(lenght))

    # Itero sobre el Dataset y lo fragmento
    for x in range(lenght):
        json_line = dataset.at[x, 'lines']  # Leo cada JSON LINE
        input_data[json_line['bill_id']] = json_line['text']  # Agrego el INPUT al Dict
        target_data[json_line['bill_id']] = json_line['summary']  # Agrego el TARGET al Dict

        # input_text = __replace_semicolon(json_line['text'], 10)
        input_text = json_line['text']
        cleaned_text = __clean_text(input_text)  # Limpieza del fallo. TEXT_INPUT es lo que se le envia a NLP.

        # Actualizo el Dict con los inputs preprocesados
        input_data[json_line['bill_id']] = cleaned_text

    # Guardado
    with open('./outputs/preprocessed_input.json', 'w', encoding='utf8') as outfile:
        json.dump(input_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    with open('./outputs/targets.json', 'w', encoding='utf8') as outfile:
        json.dump(target_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    return input_data


def __test():

    pattern = r"\b[0-9]{1,2}(?:\.[0-9]{3})+\b"

    numbers = [
        '23.756',
        '3.453',
        '556',
        '1.658.239'
    ]

    for s in numbers:
        print(re.sub(pattern, lambda x: x.group().replace(".", ""), s))

