import matplotlib.pyplot as plt


def print_rouge(scores):

    x1 = []
    x2 = []
    xL = []
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

    # plotting the rouge-1 points
    plt.plot(x1, y1, label="rouge-1", marker='o', markersize=8)

    # plotting the rouge-2 points
    x2 = x1
    plt.plot(x2, y2, label="rouge-2", marker='o', markersize=8)

    # plotting the rouge-2 points
    xL = x1
    plt.plot(xL, yL, label="rouge-L", marker='o', markersize=8)

    # setting y axis range
    plt.ylim(0, 1)

    # naming the x axis
    plt.xlabel('rouge score')
    # naming the y axis
    plt.ylabel('document number')

    # giving a title to my graph
    plt.title('ROUGE Recall Scores')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()


