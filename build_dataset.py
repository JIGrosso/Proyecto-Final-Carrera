from random import randint
import json


def main():
    dataset = {'lines': []}
    lines = []
    read = []
    counts = 0

    while True:
        text_id = randint(309, 15000)
        if text_id not in read:
            try:
                with open('./dataset/fallos_clasificados/' + str(text_id) + '.json', 'r', encoding='utf8') as file:
                    fallo = json.load(file)
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
        # Setear la cantidad de fallos a leer.
        if counts == 10000:
            break
    dataset['lines'] = lines
    # Guardado - Setear el nombre del archivo en donde se van a almacenar
    with open('./dataset/test_input_dataset.json', 'w', encoding='utf8') as outfile:
        json.dump(dataset, outfile, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()
