import json
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize


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
    refined_text = refined_text.replace('Dras.', 'doctoras')
    refined_text = refined_text.replace('Sr.', 'señor')
    refined_text = refined_text.replace('Sra.', 'señora')
    refined_text = refined_text.replace('Sres.', 'señores')
    refined_text = refined_text.replace('Sras.', 'señoras')
    refined_text = refined_text.replace('Excma.', 'excelentisima')
    refined_text = refined_text.replace('Excmo.', 'excelentisimo')
    refined_text = refined_text.replace('pag.', 'página')
    refined_text = refined_text.replace('inc.', 'inciso')

    refined_text = refined_text.replace("f.", "foja")
    refined_text = refined_text.replace("fs.", "fojas")
    refined_text = refined_text.replace("1er.", "primer")
    refined_text = refined_text.replace("1ro.", "primero")
    refined_text = refined_text.replace("1ero.", "primero")
    refined_text = refined_text.replace("1era.", "primera")
    refined_text = refined_text.replace("1ra.", "primera")
    refined_text = refined_text.replace("2do.", "segundo")
    refined_text = refined_text.replace("2da.", "segunda")
    refined_text = refined_text.replace("4to.", "cuarto")
    refined_text = refined_text.replace("4ta.", "cuarta")
    refined_text = refined_text.replace("5to.", "quinto")
    refined_text = refined_text.replace("5ta.", "quinta")
    refined_text = refined_text.replace("ss.", "siguientes")
    refined_text = refined_text.replace("sgtes.", "siguientes")
    refined_text = refined_text.replace("vta.", "vuelta")
    refined_text = refined_text.replace("C.N.Civ.", "Código Nacional Civil")
    refined_text = refined_text.replace("C. N. Civ.", "Código Nacional Civil")
    refined_text = refined_text.replace("C. N.", "Código Nacional")
    refined_text = refined_text.replace("C.N.", "Código Nacional")
    refined_text = refined_text.replace("Civ.", "Civil")
    refined_text = refined_text.replace("L.O.", "Ley Orgánica") #chequear esta
    refined_text = refined_text.replace("Nro.", "Número")
    refined_text = refined_text.replace("cctes.", "consecuentes")
    refined_text = refined_text.replace("D.L.", "Decreto Ley")
    refined_text = refined_text.replace("D. L.", "Decreto Ley")
    refined_text = refined_text.replace("L.C.T.", "Ley del Contrato del Trabajo")
    refined_text = refined_text.replace("L. C. T.", "Ley del Contrato del Trabajo")
    refined_text = refined_text.replace("Ed.", "Edición")
    refined_text = refined_text.replace("ed.", "edición")
    refined_text = refined_text.replace("E.D.", "E D")
    refined_text = refined_text.replace("L.L.", "L L")
    refined_text = refined_text.replace("pág.", "página")
    refined_text = refined_text.replace("Const.", "Constitución")
    refined_text = refined_text.replace("Avda.", "Ávenida")
    refined_text = refined_text.replace("R.J.N.", "Reglamento para la Justicia Nacional")
    refined_text = refined_text.replace("Cód.", "Código")
    refined_text = refined_text.replace("Cod.", "Código")
    refined_text = refined_text.replace("C. P.", "Código Penal")
    refined_text = refined_text.replace("C.P.", "Código Penal")
    refined_text = refined_text.replace("C.P.P.", "Código Procesal Penal")
    refined_text = refined_text.replace("C. P. P.", "Código Procesal Penal")
    refined_text = refined_text.replace("C.P.C.", "Código Procesal Constitucional")
    refined_text = refined_text.replace("C. P. C.", "Código Procesal Constitucional")
    refined_text = refined_text.replace("Bs.", "Buenos")
    refined_text = refined_text.replace("As.", "Aires")
    refined_text = refined_text.replace("Bs.As.", "Buenos Aires")
    refined_text = refined_text.replace("ob.", "obra")
    refined_text = refined_text.replace("cit.", "citada")
    refined_text = refined_text.replace("Prov.", "Provincial")
    refined_text = refined_text.replace("Nac.", "Nacional")
    refined_text = refined_text.replace("Expte.", "Expediente")
    refined_text = refined_text.replace("Direc.", "Dirección")
    refined_text = refined_text.replace("Art.", "Artículo")
    refined_text = refined_text.replace("Fdo.", "Firmado")
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
    # refined_text = FIX_START_re.sub('', refined_text)
    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent
    # refined_text = FIX_PERIOD_re.sub(". \g<1>", refined_text)

    return refined_text


def __remove_stop_words(input_text):

    # We only want to work with lowercase for the comparisons
    # text = input_text.lower()
    text = input_text  # Utilizar mayúsculas hace que el PoS de mejores resultados.

    # remove punctuation and split into seperate words
    words = re.findall(r'\w+', text, flags=re.UNICODE)  # | re.LOCALE)

    # split into separate words with punctuation
    words = text.split()

    # This is the simple way to remove stop words

    important_words = []
    for word in words:
        if word.lower() not in stopwords.words('spanish'):
            important_words.append(word)

    nsw_text = " "

    return nsw_text.join(important_words)

    # This is the more pythonic way
    # important_words = filter(lambda x: x not in stopwords.words('spanish'), words)


def __split_into_sentences(paragraph):
    return sent_tokenize(paragraph)


def __split_input(text):
    '''
        Divide el input en párrafos haciendo uso de "/r/r/n".
        Luego limpia con __clean_text() cada division.
    '''
    # Split en parrafos
    paragraphs = text.split("\r\r\n")
    cleaned_paragraphs = []
    cleaned_sentences = []
    for p in paragraphs:
        # if len(p.split()) > 3:  # Elimina los párrafos de menos de 3 palabras.
        cleaned_paragraph = __clean_text(p)
        nsw_paragraph = __remove_stop_words(cleaned_paragraph)
        # nsw_paragraph = cleaned_paragraph
        # Loop para chequear que oraciones sean mayor a 3 palabras, y pasarlas a un arreglo que luego se hace append a cleaned_paragrahps

        aux_splitted_sentences = __split_into_sentences(nsw_paragraph)
        splitted_sentences = []

        for ss in aux_splitted_sentences:
            # if len(sp.split()) > 3:
            splitted_sentences.append(ss)
            cleaned_sentences.append(ss)

        cleaned_paragraphs.append(splitted_sentences)  # Agrego array de oraciones a array de parrafos

    '''
    Loop through each cleaned paragraph, and through each sentence of it 
    (lo dejo armado para aplicar limpieza o por si hay que hacer algo a cada oracion en particular)
  
    for cp in cleaned_paragraphs:
        #print('cleaned paragraph: ')
        #print(cp)
        for cs in cp:
            print('cleaned sentence: ')
            print(cs)
    '''

    return cleaned_paragraphs, cleaned_sentences


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

        # Actualizo el Dict con los inputs preprocesados
        input_data[json_line['bill_id']] = cleaned_text  # Agrego el INPUT al Dict. CLEANED_TEXT es lo que se le envia a NLP.
        splitted_input_data[json_line['bill_id']] = splitted_text

    # Guardado
    with open('./outputs/preprocessed_input.json', 'w', encoding='utf8') as outfile:
        json.dump(input_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    with open('./outputs/preprocessed_splitted_input.json', 'w', encoding='utf8') as outfile:
        json.dump(splitted_input_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)
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

