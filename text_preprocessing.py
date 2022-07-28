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
# DASH_RE = re.compile('--+')
BAD_PUNCT_re = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
WHITESPACE_re = re.compile('\s+')
EMPTY_SENT_re = re.compile('[,\.]\ *[\.,]')
FIX_START_re = re.compile('^[^A-Za-z]*')


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
    refined_text = refined_text.replace("L.O.", "Ley Orgánica")  # chequear esta
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
    refined_text = refined_text.replace("E.D.", "el Derecho")
    refined_text = refined_text.replace("L.L.", "la ley")
    refined_text = refined_text.replace("cfr.", "conforme")

    # Elimina los puntos entre números. Ejemplo : 16.233 -> 16233
    refined_text = re.sub(DOT_BETWEEN_NUMBERS_re, lambda x: x.group().replace(".", ""), refined_text)

    # Remove annoying punctuation, that's not relevant
    refined_text = BAD_PUNCT_re.sub('', refined_text)

    # removing newlines, tabs, and extra spaces.
    refined_text = WHITESPACE_re.sub(' ', refined_text)

    # If we ended up with "empty" sentences - get rid of them.
    refined_text = EMPTY_SENT_re.sub('.', refined_text)

    # TODO Reescribir o eliminar
    # Fix weird period issues + start of text weirdness
    # text = re.sub('\.(?=[A-Z])', '  . ', text)

    # Get rid of anything thats not a word from the start of the text
    refined_text = FIX_START_re.sub('', refined_text)

    return refined_text


def __remove_stop_words(input_text):

    text = input_text  # Utilizar mayúsculas hace que el PoS de mejores resultados.

    # remove punctuation and split into seperate words
    # words = re.findall(r'\w+', text, flags=re.UNICODE)  # | re.LOCALE)

    # split into separate words with punctuation
    words = text.split()

    # This is the simple way to remove stop words

    important_words = []
    for word in words:
        if word.lower() not in stopwords.words('spanish'):
            important_words.append(word)

    nsw_text = " "

    return nsw_text.join(important_words)


def __split_input(paragraphs):

    paragraphs_into_sentences = []  # Cada elemento de este array es un array de oraciones.
    original_sentences = []  # Cada elemento de este array es una oracion que no se le quitaran las SW.

    for p in paragraphs:
        aux_splitted_sentences = sent_tokenize(p)
        sentences = []  # Inicializa vacío, se insertan todas las oraciones del párrafo, y luego se se inserta en array de párrafos

        for ss in aux_splitted_sentences:
            if len(ss.split()) > 2:
                original_sentences.append(ss)
                sentences.append(ss)
        if len(sentences) > 0:
            paragraphs_into_sentences.append(sentences)  # Agrego array de oraciones a array de parrafos

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

    return paragraphs_into_sentences, original_sentences


def process(dataset):
    # Me sirve para ver la estructura del dataset - returns Panda's object
    # dataset = pd.read_json('./dataset/'+args.filename+'.json')

    # Auxiliares
    input_data = {}  # Dict con todos los inputs. Se utiliza para la tecnica de TextRank
    splitted_input_data = {}  # Dict con todos los inputs separados en párrafos. Se utiliza para la tecnica de deep learning

    target_data = {}  # Dict con todos los targets
    index = dataset.index  # Longitud del dataset
    lenght = len(index)  # Longitud del dataset
    print("Cantidad de documentos legales: " + str(lenght))

    # Itero sobre el Dataset
    for x in range(lenght):
        json_line = dataset.at[x, 'lines']  # Leo cada JSON LINE

        target_data[json_line['bill_id']] = __clean_text(json_line['summary'])  # Agrego el TARGET al Dict

        text = json_line['text']
        # Split en parrafos
        if "\r\r\n" in text:
            paragraphs = text.split("\r\r\n")
        elif "\r\n" in text:
            paragraphs = text.split("\r\n")
        else:
            paragraphs = text.split("\n")

        # paragraphs = text.split("\r\r\n")
        cleaned_paragraphs = []
        cleaned_text = ''  # Input para TextRank

        for p in paragraphs:
            if len(p.split()) > 3:  # Elimina los párrafos de menos de 3 palabras.
                cleaned_paragraphs.append(__clean_text(p))
                cleaned_text = cleaned_text + __clean_text(p) + "\n"

        paragraphs_into_sentences, original_sentences = __split_input(cleaned_paragraphs)

        nsw_paragraphs_into_sentences = []  # Input para TextFeatures
        nsw_sentences = []  # Input para TextFeatures
        splitted_text = []  # Agrupo los Inputs.

        for cp in paragraphs_into_sentences:  # Recorro cada array de arrays
            aux_nsw_cp = []  # Auxiliar vacío para armar parrafos

            for cs in cp:  # Recorro cada elemento (oracion) del array
                nsw_cs = __remove_stop_words(cs)  # Elimino stop words
                if len(nsw_cs.split()) > 0:
                    nsw_sentences.append(nsw_cs)  # Almaceno oraciones sin stop words
                    aux_nsw_cp.append(nsw_cs)  # Armo parrafo de oraciones sin stop words
                else:
                    original_sentences.remove(cs)

            nsw_paragraphs_into_sentences.append(aux_nsw_cp)  # Almaceno parrafos con oraciones sin stop words

        # En nsw_paragraphs_into_sentences tenemos un array de párrafos, donde cada uno es un array de oraciones sin stop words
        # En nsw_sentences tenemos todas las oraciones sin stop words, sin dividir en párrafos
        if len(nsw_sentences) == len(original_sentences):
            splitted_text.append(nsw_paragraphs_into_sentences)
            splitted_text.append(nsw_sentences)
            splitted_text.append(original_sentences)
        else:
            print("An error ocurred when preprocessing the document " + json_line['bill_id'])

        # Actualizo el Dict con los inputs preprocesados
        input_data[json_line['bill_id']] = cleaned_text
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

