import pandas as pd
import json
import rouge_script
import plots


def analyze_input(filename):

    dataset = pd.read_json('./dataset/' + filename + '.json')

    input_data = {}  # Dict con todos los inputs
    output_data = {}  # Dict con todos los targets
    length_data = {}
    index = dataset.index  # Longitud del dataset
    lenght = len(index)  # Longitud del dataset
    print("Cantidad de documentos legales: " + str(lenght))

    # Itero sobre el Dataset y lo fragmento
    for x in range(lenght):
        json_line = dataset.at[x, 'lines']
        input_data[json_line['bill_id']] = json_line['text']
        output_data[json_line['bill_id']] = json_line['summary']
        # Length analyzer
        length_data[json_line['bill_id']] = float(len(json_line['summary'])/len(json_line['text']))


    # Guardado
    with open('./outputs/input_analyzer.json', 'w', encoding='utf8') as outfile:
        json.dump(input_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    with open('./outputs/output_analyzer.json', 'w', encoding='utf8') as outfile:
        json.dump(output_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)
    with open('./outputs/length_analyzer.json', 'w', encoding='utf8') as outfile:
        json.dump(length_data, outfile, indent=4, sort_keys=True, ensure_ascii=False)

    rouge_scores = rouge_script.get_rouge_scores('input_analyzer', 'output_analyzer')
    # Si el recall es alto quiere decir que gran parte de las palabras del target estan contenidas en el input
    plots.print_rouge_recall(rouge_scores)
    plots.print_rouge_precision(rouge_scores)


def main():
    analyze_input('test_input_dataset')


if __name__ == "__main__":
    main()


