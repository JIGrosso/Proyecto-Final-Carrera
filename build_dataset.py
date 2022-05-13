from random import randint
import json


def main():
    dataset = {'lines': []}
    lines = []

    for i in range(0, 16000):
        text_id = randint(309, 15000)
        try:
            with open('./dataset/fallos_clasificados/' + str(text_id) + '.json', 'r', encoding='utf8') as file:
                fallo = json.load(file)
            text_line = {
                         'bill_id': fallo['id_fallo'],
                         'text': fallo['texto_fallos'],
                         'summary': fallo['texto_sumario']
                        }
            lines.append(text_line)
        except:
            print('El archivo ' + str(text_id) + '.json no existe')

    dataset['lines'] = lines
    # Guardado
    with open('./dataset/fallos_clasificados.json', 'w', encoding='utf8') as outfile:
        json.dump(dataset, outfile, indent=4, sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    main()
