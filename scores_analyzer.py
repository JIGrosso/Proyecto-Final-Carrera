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

    # print(f"Rouge-1: F-Score = {total_rouge_1_f / total}")
    # print(f"Rouge-1: Precision = {total_rouge_1_p/total}")
    # print(f"Rouge-1: Recall = {total_rouge_1_r / total}")

    # print(f"Rouge-2: F-Score = {total_rouge_2_f / total}")
    # print(f"Rouge-2: Precision = {total_rouge_2_p / total}")
    # print(f"Rouge-2: Recall = {total_rouge_2_r / total}")

    # print(f"Rouge-L: F-Score = {total_rouge_l_f / total}")
    # print(f"Rouge-L: Precision = {total_rouge_l_p / total}")
    # print(f"Rouge-L: Recall = {total_rouge_l_r / total}")

    print(f"{total_rouge_1_f / total}")
    print(f"{total_rouge_1_p/total}")
    print(f"{total_rouge_1_r / total}")

    print(f"{total_rouge_2_f / total}")
    print(f"{total_rouge_2_p / total}")
    print(f"{total_rouge_2_r / total}")

    print(f"{total_rouge_l_f / total}")
    print(f"{total_rouge_l_p / total}")
    print(f"{total_rouge_l_r / total}")


def scores_ranges(filename):

    dataset = pd.read_json('./outputs/' + filename + '.json')

    total = 0
    total_recall_in_range = 0

    # total_rouge_1_r = 0
    # total_rouge_2_r = 0
    # total_rouge_l_r = 0

    bottom = 0.6
    top = 1.0

    # Itero sobre el Dataset y lo fragmento
    for key, value in dataset.items():
        total += 1

        for summary_scores in value:

            condition_rouge_1 = bottom <= round(summary_scores['rouge-1']['r'], 3) <= top

            condition_rouge_2 = bottom <= round(summary_scores['rouge-2']['r'], 3) <= top

            condition_rouge_l = bottom <= round(summary_scores['rouge-l']['r'], 3) <= top
            # if condition_rouge_1 and condition_rouge_2 and condition_rouge_l:
            if condition_rouge_l:
                # print(key)
                # print(f"1:{round(summary_scores['rouge-1']['r'], 3)}")
                # print(f"2: {round(summary_scores['rouge-1']['r'], 3)}")
                # print(f"L: {round(summary_scores['rouge-l']['r'], 3)}")
                total_recall_in_range += 1

    print(f"Porcentaje: %{round(total_recall_in_range*100/total, 2)}")
    print(f"Numero de documentos: {total_recall_in_range}")

    # print(f"[ROUGE-1] Porcentaje de documentos con recall >= {bottom} y < {top}: %{round(total_rouge_1_r*100/total, 2)}")
    # print(f"[ROUGE-2] Porcentaje de documentos con recall >= {bottom} y < {top}: %{round(total_rouge_2_r * 100 / total, 2)}")
    # print(f"[ROUGE-L] Porcentaje de documentos con recall >= {bottom} y < {top}: %{round(total_rouge_l_r*100/total, 2)}")


def main():
    # analyze_scores('text_rank_summaries_rouge_scores')
    # analyze_scores('tf_dl_summaries_rouge_scores')
    # analyze_scores('input_analyzer_16390_rouge_scores')
    # analyze_scores('input_analyzer_1124571_rouge_scores')
    # analyze_scores('input_analyzer_1167175_rouge_scores')
    # analyze_scores('input_analyzer_1193678_rouge_scores')
    # analyze_scores('input_analyzer_1231993_rouge_scores')
    # analyze_scores('input_analyzer_1234937_rouge_scores')

    scores_ranges('Lote 1 90%/Part 1/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 1 90%/Part 1/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 1 90%/Part 2/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 1 90%/Part 2/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 1 90%/Part 3/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 1 90%/Part 3/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 1 90%/Part 4/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 1 90%/Part 4/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 1/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 1/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 2/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 2/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 3/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 3/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 4/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 2 90%/Part 4/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 1/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 1/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 2/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 2/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 3/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 3/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 4/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 3 90%/Part 4/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 1/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 1/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 2/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 2/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 3/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 3/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 4/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 4 90%/Part 4/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 1/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 1/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 2/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 2/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 3/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 3/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 4/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 5 90%/Part 4/tf_dl_summaries_rouge_scores')
    scores_ranges('Lote 6 90%/text_rank_summaries_rouge_scores')
    scores_ranges('Lote 6 90%/tf_dl_summaries_rouge_scores')
    # scores_ranges('input_analyzer_16390_rouge_scores')
    # scores_ranges('input_analyzer_1124571_rouge_scores')
    # scores_ranges('input_analyzer_1167175_rouge_scores')
    # scores_ranges('input_analyzer_1193678_rouge_scores')
    # scores_ranges('input_analyzer_1231993_rouge_scores')
    # scores_ranges('input_analyzer_1234937_rouge_scores')


if __name__ == "__main__":
    main()
