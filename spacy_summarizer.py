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
    """
    Borrowed from the FNDS text processing with additional logic added in.
    Note: we do not take care of token breaking - assume SPACY's tokenizer
    will handle this for us.
    """

    # Indicate section headers, we need them for features
    text = SECTION_HEADER_RE.sub('SECTION-HEADER', text)
    # For simplicity later, remove '.' from most common acronym
    text = text.replace("U.S.", "US")
    text = text.replace('SEC.', 'Section')
    text = text.replace('Sec.', 'Section')
    text = USC_re.sub('USC', text)

    # Remove parantheticals because they are almost always references to laws
    # We could add a special tag, but we just remove for now
    # Note we dont get rid of nested parens because that is a complex re
    # text = PAREN_re.sub('LAWREF', text)
    text = PAREN_re.sub('', text)

    # Get rid of enums as bullets or ` as bullets
    text = BULLET_RE.sub(' ', text)

    # Clean html
    text = text.replace('&lt;all&gt;', '')

    # Remove annoying punctuation, that's not relevant
    text = BAD_PUNCT_RE.sub('', text)

    # Get rid of long sequences of dashes - these are formating
    text = DASH_RE.sub(' ', text)

    # removing newlines, tabs, and extra spaces.
    text = WHITESPACE_RE.sub(' ', text)

    # If we ended up with "empty" sentences - get rid of them.
    text = EMPTY_SENT_RE.sub('.', text)

    # Attempt to create sentences from bullets
    text = replace_semicolon(text)

    # Fix weird period issues + start of text weirdness
    # text = re.sub('\.(?=[A-Z])', '  . ', text)
    # Get rid of anything thats not a word from the start of the text
    text = FIX_START_RE.sub('', text)
    # Sometimes periods get formatted weird, make sure there is a space between periods and start of sent
    text = FIX_PERIOD.sub(". \g<1>", text)

    # Fix quotes
    text = text.replace('``', '"')
    text = text.replace('\'\'', '"')

    # Add special punct back in
    text = text.replace('SECTION-HEADER', '<SECTION-HEADER>')

    return text


# LIMPIEZA DE TEXTO
def text_preprocessing(text_input):
    text_input = replace_semicolon(text_input, 10)

    text_input = clean_text(text_input)

    return text_input

# MAIN

nlp = spacy.load("es_core_news_lg")
nlp_sentencizer = Spanish()

nlp.add_pipe("textrank")
nlp_sentencizer.add_pipe("sentencizer")

# Expresiones regulares

USC_re = re.compile('[Uu]\.*[Ss]\.*[Cc]\.]+')
PAREN_re = re.compile('\([^(]+\ [^\(]+\)')
BAD_PUNCT_RE = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE)
BULLET_RE = re.compile('\n[\ \t]*`*\([a-zA-Z0-9]*\)')
DASH_RE = re.compile('--+')
WHITESPACE_RE = re.compile('\s+')
EMPTY_SENT_RE = re.compile('[,\.]\ *[\.,]')
FIX_START_RE = re.compile('^[^A-Za-z]*')
FIX_PERIOD = re.compile('\.([A-Za-z])')
SECTION_HEADER_RE = re.compile('SECTION [0-9]{1,2}\.|\nSEC\.* [0-9]{1,2}\.|Sec\.* [0-9]{1,2}\.')

dataset = pd.read_json('./dataset/dataset-spa-test3.json')

targets = []

# Auxiliares
index = dataset.index
lenght = len(index)
print("Cantidad de documentos legales: " + str(lenght))

data = {}

for x in range(lenght):
  aux_line = dataset.at[x, 'lines']
  print('FALLO: ' + aux_line['bill_id'] + '\n' + aux_line['text'] + '\n')
  data['fallo '+ aux_line['bill_id']] = aux_line['text']
  print('SUMARIO: ' + aux_line['summary'] + '\n')
  data['sumario ' + aux_line['bill_id']] = aux_line['summary']
  #Sentencizer
  aux_doc = nlp_sentencizer(aux_line['summary'])
  targets.append(aux_doc)
  text = aux_line['text']

print('----------------- \n')
print(text)

with open('data.txt', 'w') as outfile:
    json.dump(data, outfile, indent=4)