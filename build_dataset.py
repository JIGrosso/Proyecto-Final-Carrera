from random import randint
import json
import pandas

for i in range(0, 10):
    text_id = randint(309, 15000)
    with open('./dataset/fallos_clasificados/' + str(text_id) + '.json', 'r', encoding='utf8') as file:
        fallo = json.load(file)
    print(fallo['id_fallo'])
    print(fallo['texto_fallos'])
    print(fallo['texto_sumario'])
