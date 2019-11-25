import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os
import json

style.use('ggplot')


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


folder = "/home/bt/Documents/Studia/DLR/ReinforcementLearnig/qtable_charts"

for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(e)

fig = plt.figure(figsize=(12, 9))

data = {}

with open('hiperparameters.json') as fp:
    data = json.load(fp)

for i in range(0, data['episodes'], 10):
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    q_table = np.load("qtables/{}-qtable.npy".format(i))

    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(x,
                        y,
                        c=get_q_color(y_vals[0], y_vals)[0],
                        marker="o",
                        alpha=get_q_color(y_vals[0], y_vals)[1])
            ax2.scatter(x,
                        y,
                        c=get_q_color(y_vals[1], y_vals)[0],
                        marker="o",
                        alpha=get_q_color(y_vals[1], y_vals)[1])
            ax3.scatter(x,
                        y,
                        c=get_q_color(y_vals[2], y_vals)[0],
                        marker="o",
                        alpha=get_q_color(y_vals[2], y_vals)[1])

            ax1.set_title("Action 0")
            ax2.set_title("Action 1")
            ax3.set_title("Action 2")
            ax3.set_xlabel("Height")
            ax3.set_ylabel("Velocity")

    plt.savefig("qtable_charts/{}.png".format(i))
    plt.clf()
