import argparse
import json
import pandas as pd
import text_preprocessing
import spacy_summarizer

if __name__ == "__main__":

    print("Reading Text...")
    print("Preprocessing Text...")
    print("Applying Summarizer...")
    print("Process finished...")

    parser = argparse.ArgumentParser(description='Parser for PyTextRank parameters')
    parser.add_argument('--filename', metavar='path', required=True, help='the name of the input file')

    args = parser.parse_args()

    dataset = pd.read_json('./dataset/' + args.filename + '.json')

    # TODO Separar input del target. Acá o en text_preprocessing.

    preprocessed_text = text_preprocessing.process(dataset)

    with open('./outputs/preprocessed_text.json', 'w', encoding='utf8') as outfile:
        json.dump(preprocessed_text, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    spacy_summarizer.summary(preprocessed_text)

    # Comparar oración a oración. Dejo por las dudas
    """
    for a_sent in outputs[6].sents:
        a = str(a_sent).lower()

        for b_sent in targets[6].sents:
            b = str(b_sent).lower()
            print('Oración a: ' + str(a))
            print('Oración b: ' + str(b) + '\n')
            print('Comparación: ')
            seq = dl.SequenceMatcher(None, a.lower(), b.lower())
            d = seq.ratio() * 100
            print(str(d) + '\n')

            print('Matcheos: \n')

            matches = dl.SequenceMatcher(None, a, b).get_matching_blocks()
            for match in matches:
                print(a[match.a:match.a + match.size])
    """