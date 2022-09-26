import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_scores(x1, y1, y2, yL, score_type):

    # plotting the rouge-1 points
    plt.plot(x1, y1, label="rouge-1", marker='o', markersize=5)

    # plotting the rouge-2 points
    x2 = x1
    plt.plot(x2, y2, label="rouge-2", marker='o', markersize=5)

    # plotting the rouge-L points
    xL = x1
    plt.plot(xL, yL, label="rouge-L", marker='o', markersize=5)

    # setting y axis range
    plt.ylim(0, 1)

    # naming the x axis
    plt.xlabel('document number')
    # naming the y axis
    plt.ylabel('rouge score')

    # giving a title to my graph
    plt.title('ROUGE '+score_type+' Scores')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()


def scatter_scores(x1, y, rouge_type, score_type):

    sizes = np.random.uniform(30, 30, len(x1))
    colors = np.random.uniform(15, 80, len(x1))

    plt.scatter(x1, y, s=sizes, c=colors)

    plt.ylim([0.0, 1.0])

    # naming the x axis
    plt.xlabel('documents')
    # naming the y axis
    plt.ylabel('rouge score')

    # giving a title to my graph
    plt.title('ROUGE ' + rouge_type + ' - ' + score_type + ' Scores')

    plt.show()


def bar_scores(x, y, technique_type, score_type, rouge_type, color):

    sizes = np.random.uniform(30, 30, len(x))
    colors = np.random.uniform(15, 80, len(x))

    plt.bar(x, y, width=0.5, edgecolor="white", linewidth=0.7, color=color)

    plt.ylim([0.0, 1.0])

    # naming the x axis
    plt.xlabel('Document number')
    # naming the y axis
    plt.ylabel('Rouge score')

    # giving a title to my graph
    plt.title(technique_type + ' - ROUGE ' + score_type + ' - ' + rouge_type + ' Scores')

    name = technique_type.lower().replace(' ', '_') + '_' + rouge_type.lower() + '_rouge_' + score_type

    plt.savefig('./results/' + name + '.png')

    plt.show()


def print_rouge_recall(scores, technique):

    x1 = []
    y1 = []
    y2 = []
    yL = []
    x_index = 0

    for text_id in scores:
        # x axis values
        x1.append(x_index)
        # corresponding y axis values
        y1.append(scores[text_id][0]['rouge-1']['r'])
        y2.append(scores[text_id][0]['rouge-2']['r'])
        yL.append(scores[text_id][0]['rouge-l']['r'])
        x_index += 1

    # scatter_scores(x1, y1, '1', 'Recall')
    # scatter_scores(x1, y2, '2', 'Recall')
    # scatter_scores(x1, yL, 'L', 'Recall')
    bar_scores(x1, y1, technique, '1', 'Recall', 'steelblue')
    bar_scores(x1, y1, technique, '2', 'Recall', 'slateblue')
    bar_scores(x1, yL, technique, 'L', 'Recall', 'yellowgreen')


def print_rouge_precision(scores, technique):

    x1 = []
    y1 = []
    y2 = []
    yL = []
    x_index = 0

    for text_id in scores:
        # x axis values
        x1.append(x_index)
        # corresponding y axis values
        y1.append(scores[text_id][0]['rouge-1']['p'])
        y2.append(scores[text_id][0]['rouge-2']['p'])
        yL.append(scores[text_id][0]['rouge-l']['p'])
        x_index += 1

    # scatter_scores(x1, yL, 'L', 'Precision')
    bar_scores(x1, y1, technique, '1', 'Precisión', 'steelblue')
    bar_scores(x1, y2, technique, '2', 'Precisión', 'slateblue')
    bar_scores(x1, yL, technique, 'L', 'Precisión', 'yellowgreen')


def print_rouge_f1_score(scores, technique):

    x1 = []
    y1 = []
    y2 = []
    yL = []
    x_index = 0

    for text_id in scores:
        # x axis values
        x1.append(x_index)
        # corresponding y axis values
        y1.append(scores[text_id][0]['rouge-1']['f'])
        y2.append(scores[text_id][0]['rouge-2']['f'])
        yL.append(scores[text_id][0]['rouge-l']['f'])
        x_index += 1

    bar_scores(x1, y1, technique, '1', 'F-Score', 'steelblue')
    bar_scores(x1, y2, technique, '2', 'F-Score', 'slateblue')
    bar_scores(x1, yL, technique, 'L', 'F-Score', 'yellowgreen')


def print_scores_from_file(filename):

    dataset = pd.read_json(f'./outputs/{filename}.json', typ='series')

    scores = {}

    for key, value in dataset.items():
        scores[key] = value

    print_rouge_recall(scores, 'Recall')
    print_rouge_precision(scores, 'Precision')
    print_rouge_f1_score(scores, 'F-Score')


def main():
    # print_scores_from_file('input_analyzer_16390_rouge_scores')
    # print_scores_from_file('input_analyzer_1124571_rouge_scores')
    # print_scores_from_file('input_analyzer_1167175_rouge_scores')
    # print_scores_from_file('input_analyzer_1193678_rouge_scores')
    # print_scores_from_file('input_analyzer_1231993_rouge_scores')
    # print_scores_from_file('input_analyzer_1234937_rouge_scores')

    print_scores_from_file("tf_dl_summaries_rouge_scores")


if __name__ == "__main__":
    main()


