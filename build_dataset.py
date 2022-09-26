from random import randint
import json
import pandas as pd


def build_random_dataset():
    # Numero de documentos
    limit = 60

    dataset = {'lines': []}
    lines = []
    read = []
    counts = 0

    while True:
        text_id = randint(309, 1500000)
        if text_id not in read:
            try:
                with open('./dataset/fallos_clasificados/' + str(text_id) + '.json', 'r', encoding='utf8') as file:
                    fallo = json.load(file)
                if len(fallo['texto_fallos']) > len(fallo['texto_sumario']):
                    text_line = {
                        'bill_id': fallo['id_fallo'],
                        'text': fallo['texto_fallos'],
                        'summary': fallo['texto_sumario']
                    }
                    lines.append(text_line)
                    counts += 1
            except:
                print('El archivo ' + str(text_id) + '.json no existe')
            read.append(text_id)
        if counts == limit:
            break
    dataset['lines'] = lines

    with open('dataset/final_test_set_' + str(limit) + '.json', 'w', encoding='utf8') as outfile:
        json.dump(dataset, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def build_ordered_dataset():
    # Numero de documentos
    limit = 10000
    dataset = {'lines': []}
    lines = []
    read = []
    counts = 0

    text_id = 1231992  # Arranca con este ID

    while True:
        text_id += 1
        if text_id not in read:
            try:
                with open('./dataset/fallos_clasificados/' + str(text_id) + '.json', 'r', encoding='utf8') as file:
                    fallo = json.load(file)
                if len(fallo['texto_fallos']) > len(fallo['texto_sumario']):
                    text_line = {
                        'bill_id': fallo['id_fallo'],
                        'text': fallo['texto_fallos'],
                        'summary': fallo['texto_sumario']
                    }
                    lines.append(text_line)
                    counts += 1
                else:
                    print(f'El archivo {text_id}.json es erróneo')
            except:
                print('El archivo ' + str(text_id) + '.json no existe')
            read.append(text_id)
        if counts == limit:
            break
    dataset['lines'] = lines

    with open(f'dataset/input_analyzer_{text_id}.json', 'w', encoding='utf8') as outfile:
        json.dump(dataset, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def build_dataset_from_scores(scores_filename):
    output_filename = f'segmented_dataset_{scores_filename}'
    output_filename = 'lote_06_90%'

    dataset = {'lines': []}
    lines = []
    counts = 0

    scores = pd.read_json('./outputs/' + scores_filename + '.json')

    for key, value in scores.items():
        for summary_scores in value:
            recall_condition_1 = 0.90 <= round(summary_scores['rouge-1']['r'], 3) <= 1.0
            recall_condition_2 = 0.90 <= round(summary_scores['rouge-2']['r'], 3) <= 1.0
            recall_condition_l = 0.90 <= round(summary_scores['rouge-l']['r'], 3) <= 1.0

            if recall_condition_1 and recall_condition_2 and recall_condition_l:
                with open(f'./dataset/fallos_clasificados/{key}.json', 'r', encoding='utf8') as file:
                    fallo = json.load(file)
                text_line = {
                    'bill_id': fallo['id_fallo'],
                    'text': fallo['texto_fallos'],
                    'summary': fallo['texto_sumario']
                }
                lines.append(text_line)
                counts += 1

    dataset['lines'] = lines

    with open(f'dataset/{output_filename}.json', 'w', encoding='utf8') as outfile:
        json.dump(dataset, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    print(counts)


def split_dataset(filename, partitions):
    dataset = pd.read_json('./dataset/' + filename + '.json')

    index = dataset.index  # Longitud del dataset
    length = len(index)  # Longitud del dataset
    print("Cantidad de documentos legales: " + str(length))

    mod = length % partitions

    if mod == 0:
        size = length / partitions
    else:
        size = (length - mod) / partitions

    new_dataset = {'lines': []}
    lines = []

    counts = 0
    partition = 1
    output_filename = f"{filename}_part_0{partition}"

    print(f'Partitions: {partitions}')
    print(f'Partition size: {size}')
    print(f'Partition extra size: {size + mod}')

    for x in range(length):

        json_line = dataset.at[x, 'lines']
        lines.append(json_line)
        counts += 1

        if counts == size:
            new_dataset['lines'] = lines

            with open(f'dataset/{output_filename}.json', 'w', encoding='utf8') as outfile:
                json.dump(new_dataset, outfile, indent=4, sort_keys=True, ensure_ascii=False)

            new_dataset = {'lines': []}
            lines = []

            counts = 0
            partition += 1
            output_filename = f"{filename}_part_0{partition}"

            if mod != 0 and partition == partitions:
                size += mod


def main():
    # build_ordered_dataset()
    # build_random_dataset()
    # build_dataset_from_scores('input_analyzer_16390_rouge_scores')
    # build_dataset_from_scores('input_analyzer_1124571_rouge_scores')
    # build_dataset_from_scores('input_analyzer_1167175_rouge_scores')
    # build_dataset_from_scores('input_analyzer_1193678_rouge_scores')
    # build_dataset_from_scores('input_analyzer_1231993_rouge_scores')
    # build_dataset_from_scores('input_analyzer_1234937_rouge_scores')
    split_dataset('lote_06_90%', 4)


if __name__ == "__main__":
    main()
