from random import randint
import json


def main():
    build_ordered_dataset()
    # build_random_dataset()


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


if __name__ == "__main__":
    main()
