import pandas as pd
import json
import rouge_script
import plots


def analyze_input():
    rouge_scores = rouge_script.get_rouge_scores('output_analyzer', 'input_analyzer')
    # Si el recall es alto quiere decir que gran parte de las palabras del target estan contenidas en el input

    # Itero sobre el Dataset y lo fragmento
    for text_id in rouge_scores:
       if rouge_scores[text_id][0]['rouge-l']['r'] >= 0.9:
           print(text_id)



def main():
    analyze_input()


if __name__ == "__main__":
    main()


