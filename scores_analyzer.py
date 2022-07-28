import pandas as pd


def analyze_scores(filename):

    dataset = pd.read_json('./outputs/' + filename + '.json')

    total = 0

    total_rouge_1_f = 0
    total_rouge_1_r = 0
    total_rouge_1_p = 0

    total_rouge_2_f = 0
    total_rouge_2_r = 0
    total_rouge_2_p = 0

    total_rouge_l_f = 0
    total_rouge_l_r = 0
    total_rouge_l_p = 0
    # Itero sobre el Dataset y lo fragmento

    for key, value in dataset.items():
        total += 1
        for summary_scores in value:
            total_rouge_1_f += summary_scores['rouge-1']['f']
            total_rouge_1_r += summary_scores['rouge-1']['r']
            total_rouge_1_p += summary_scores['rouge-1']['p']

            total_rouge_2_f += summary_scores['rouge-2']['f']
            total_rouge_2_r += summary_scores['rouge-2']['r']
            total_rouge_2_p += summary_scores['rouge-2']['p']

            total_rouge_l_f += summary_scores['rouge-l']['f']
            total_rouge_l_r += summary_scores['rouge-l']['r']
            total_rouge_l_p += summary_scores['rouge-l']['p']

    print(f"Archivo: {filename}")
    print(f"Cantidad de documentos: {total}")

    print(f"Rouge-1: F-Score = {total_rouge_1_f / total}")
    print(f"Rouge-1: Precision = {total_rouge_1_p/total}")
    print(f"Rouge-1: Recall = {total_rouge_1_r / total}")

    print(f"Rouge-2: F-Score = {total_rouge_2_f / total}")
    print(f"Rouge-2: Precision = {total_rouge_2_p / total}")
    print(f"Rouge-2: Recall = {total_rouge_2_r / total}")

    print(f"Rouge-L: F-Score = {total_rouge_l_f / total}")
    print(f"Rouge-L: Precision = {total_rouge_l_p / total}")
    print(f"Rouge-L: Recall = {total_rouge_l_r / total}")


def score_90(filename):

    dataset = pd.read_json('./outputs/' + filename + '.json')

    total = 0

    total_rouge_1_r = 0
    total_rouge_l_r = 0

    # Itero sobre el Dataset y lo fragmento
    for key, value in dataset.items():
        total += 1
        for summary_scores in value:
            if 0.7 < summary_scores['rouge-1']['r'] <= 0.9:
                total_rouge_1_r += 1
            if 0.7 < summary_scores['rouge-l']['r'] <= 0.9:
                total_rouge_l_r += 1

    print(f"[ROUGE-1] Porcentaje de documentos con recall > 0.7 y <= 0.9: %{round(total_rouge_1_r*100/total, 2)}")
    print(f"[ROUGE-L] Porcentaje de documentos con recall > 0.7 y <= 0.9: %{round(total_rouge_l_r*100/total, 2)}")


def main():
    # analyze_scores('text_rank_summaries_rouge_scores')
    # analyze_scores('tf_dl_summaries_rouge_scores')
    # analyze_scores('input_analyzer_16390_rouge_scores')
    # analyze_scores('input_analyzer_1124571_rouge_scores')
    # analyze_scores('input_analyzer_1167175_rouge_scores')
    # analyze_scores('input_analyzer_1193678_rouge_scores')
    # analyze_scores('input_analyzer_1231993_rouge_scores')
    # analyze_scores('input_analyzer_1234937_rouge_scores')

    score_90('input_analyzer_16390_rouge_scores')
    score_90('input_analyzer_1124571_rouge_scores')
    score_90('input_analyzer_1167175_rouge_scores')
    score_90('input_analyzer_1193678_rouge_scores')
    score_90('input_analyzer_1231993_rouge_scores')
    score_90('input_analyzer_1234937_rouge_scores')


if __name__ == "__main__":
    main()
