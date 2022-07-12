from random import randint
import json


def main():
    # Numero de documentos
    limit = 50

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


if __name__ == "__main__":
    main()
