import matplotlib.pyplot as plt
import numpy as np


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


def scatter_scores(x1, y1, y2, yL, score_type):

    fig, ax = plt.subplots()

    sizes = np.random.uniform(15, 80, len(x1))
    colors = np.random.uniform(15, 80, len(x1))

    ax.scatter(x1, yL, s=sizes, c=colors)

    # naming the x axis
    plt.xlabel('document number')
    # naming the y axis
    plt.ylabel('rouge score')

    # giving a title to my graph
    plt.title('ROUGE ' + score_type + ' Scores')

    plt.show()


def print_rouge_recall(scores):

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

    print_scores(x1, y1, y2, yL, 'Recall')
    scatter_scores(x1, y1, y2, yL, 'Recall')


def print_rouge_precision(scores):

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

    print_scores(x1, y1, y2, yL, 'Precision')


def print_rouge_f1_score(scores):

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

    print_scores(x1, y1, y2, yL, 'F1 Score')


