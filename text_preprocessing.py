import json
import re

import pandas as pd
from nltk.corpus import stopwords


# Definición de Expresiones Regulares
# Esto permite que sean utilizadas mas tarde con otros métodos de RE.
# fs. 227/233 -> fojas 227/233 -> hojas de un expediente. También puede ser fs. vto.
# -v. cláusula séptima- ???

A_TILDE_re = re.compile('[áÁ]')
E_TILDE_re = re.compile('[éÉ]')
I_TILDE_re = re.compile('[íÍ]')
O_TILDE_re = re.compile('[óÓ]')
U_TILDE_re = re.compile('[úÚ]')
DOT_BETWEEN_NUMBERS_re = re.compile(r"\b[0-9]{1,2}(?:\.[0-9]{3})+\b")
# BULLET_RE = re.compile('\n[\ \t]*`*\([a-zA-Z0-9]*\)')
# DASH_RE = re.compile('--+')
BAD_PUNCT_re = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
WHITESPACE_re = re.compile('\s+')
EMPTY_SENT_re = re.compile('[,\.]\ *[\.,]')
FIX_START_re = re.compile('^[^A-Za-z]*')
FIX_PERIOD_re = re.compile('\.([A-Za-z])')


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

    # Documentos Legales Argentina
    refined_text = refined_text.replace("art.", "artículo")
    refined_text = refined_text.replace("arts.", "artículos")
    refined_text = refined_text.replace('Dr.', 'doctor')
    refined_text = refined_text.replace('Dra.', 'doctora')
    refined_text = refined_text.replace('Dres.', 'doctores')
    refined_text = refined_text.replace('Sr.', 'señor')
    refined_text = refined_text.replace('Sra.', 'señora')
    refined_text = refined_text.replace('Sres.', 'señores')
    refined_text = refined_text.replace('Excma.', 'excelentisima')
    refined_text = refined_text.replace('Excmo.', 'excelentisimo')
    refined_text = refined_text.replace('pag.', 'página')
    refined_text = refined_text.replace('inc.', 'inciso')

    refined_text = refined_text.replace("f.", "foja")
    refined_text = refined_text.replace("fs.", "fojas")
    # refined_text = refined_text.replace('\r\r\n', '')


    # Elimina los puntos entre números. Ejemplo : 16.233 -> 16233
    refined_text = re.sub(DOT_BETWEEN_NUMBERS_re, lambda x: x.group().replace(".", ""), refined_text)

    # TODO Reescribir o eliminar
    # Get rid of enums as bullets or ` as bullets
    # text = BULLET_RE.sub(' ', text)

    # Remove annoying punctuation, that's not relevant
    refined_text = BAD_PUNCT_re.sub('', refined_text)

    # removing newlines, tabs, and extra spaces.
    refined_text = WHITESPACE_re.sub(' ', refined_text)

    # If we ended up with "empty" sentences - get rid of them.
    refined_text = EMPTY_SENT_re.sub('.', refined_text)

    # TODO Evaluar
    # Attempt to create sentences from bullets
    # text = replace_semicolon(text)

    # TODO Reescribir o eliminar
    # Fix weird period issues + start of text weirdness
    # text = re.sub('\.(?=[A-Z])', '  . ', text)
    # Get rid of anything thats not a word from the start of the text
    refined_text = FIX_START_re.sub('', refined_text)
    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent
    refined_text = FIX_PERIOD_re.sub(". \g<1>", refined_text)

    return refined_text


def __remove_stop_words(input_text):

    # We only want to work with lowercase for the comparisons
    text = input_text.lower()

    # remove punctuation and split into seperate words
    words = re.findall(r'\w+', text, flags=re.UNICODE)  # | re.LOCALE)

    # This is the simple way to remove stop words

    important_words = []
    for word in words:
        if word not in stopwords.words('spanish'):
            important_words.append(word)

    print(important_words)

    # This is the more pythonic way
    # important_words = filter(lambda x: x not in stopwords.words('spanish'), words)


def __split_input(text):

    # Split en parrafos
    paragraphs = text.split("\r\r\n")
    cleaned_paragraphs = []
    for p in paragraphs:
        if len(p.split()) > 3:  # Elimina los párrafos de menos de 3 palabras
            cleaned_paragraphs.append(__clean_text(p))  # Aplica limpieza del texto

    return cleaned_paragraphs


def process(dataset):

    # Me sirve para ver la estructura del dataset - returns Panda's object
    # dataset = pd.read_json('./dataset/'+args.filename+'.json')

    # Auxiliares
    input_data = {}  # Dict con todos los inputs
    splitted_input_data = {}  # Dict con todos los inputs separados en párrafos. Se utiliza para la tecnica de deep learning
    target_data = {}  # Dict con todos los targets
    index = dataset.index  # Longitud del dataset
    lenght = len(index)  # Longitud del dataset
    print("Cantidad de documentos legales: " + str(lenght))

    # Itero sobre el Dataset y lo fragmento
    for x in range(lenght):
        json_line = dataset.at[x, 'lines']  # Leo cada JSON LINE

        target_data[json_line['bill_id']] = json_line['summary']  # Agrego el TARGET al Dict

        splitted_text = __split_input(json_line['text'])
        cleaned_text = __clean_text(json_line['text'])
        '''
        for paragraph in splitted_text:
            cleaned_text = cleaned_text + __clean_text(paragraph) + '/n'  # Limpieza del fallo. Agregar el '/n' hizo que mejoren los resultados.
        '''

        # Actualizo el Dict con los inputs preprocesados
        input_data[json_line['bill_id']] = cleaned_text  # Agrego el INPUT al Dict. CLEANED_TEXT es lo que se le envia a NLP.
        splitted_input_data[json_line['bill_id']] = splitted_text

    # Guardado
    with open('./outputs/preprocessed_input.json', 'w', encoding='utf8') as outfile:
        json.dump(input_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    with open('./outputs/targets.json', 'w', encoding='utf8') as outfile:
        json.dump(target_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    return input_data, splitted_input_data


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


# __remove_stop_words('test')
# __test()

