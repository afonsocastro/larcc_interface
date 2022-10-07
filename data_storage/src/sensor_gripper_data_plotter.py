#!/usr/bin/env python3
import argparse
import json
import math
import os
from matplotlib import pyplot as plt
import numpy as np
import statistics
from matplotlib.widgets import Button, Slider
from numpy import mean

parser = argparse.ArgumentParser(description="Arguments for trainning script")
parser.add_argument("-a", "--axis", type=str, default="xyz",
                    help="Chooses wihch axis to show in the plot")
parser.add_argument("-v", "--variables", type=str, default="forces",
                    help="Chooses wich variables to show in the plot")
parser.add_argument("-sm", "--smooth", type=int, default=0,
                    help="Chooses wich variables to show in the plot")

args = vars(parser.parse_args())

axis = [False, False, False]

for a in args["axis"]:
    if a == "x":
        axis[0] = True

    if a == "y":
        axis[1] = True

    if a == "z":
        axis[2] = True


def mean_values(vector, k=5):

    mean_vector = []

    for j in range(0, len(vector)):
        if j < k:
            mean_vector.append(vector[j])
        else:
            mean_number = mean(vector[j-k:j])
            mean_vector.append(mean_number)

    return mean_vector

# The function to be called anytime a slider's value changes
def update(val):
    ax.set_xlim(0, end_slider.val - init_slider.val)
    init_val = next(x for x, val in enumerate(data["timestamp"]) if val > init_slider.val)

    if axis[0]:
        linesfx.set_xdata(data["timestamp"][:-1] - data["timestamp"][init_val] * np.ones(len(data["timestamp"][:-1])))
        linesfx.set_ydata(y1[:-1])
    if axis[1]:
        linesfy.set_xdata(data["timestamp"][:-1] - data["timestamp"][init_val] * np.ones(len(data["timestamp"][:-1])))
        linesfy.set_ydata(y2[:-1])
    if axis[2]:
        linesfz.set_xdata(data["timestamp"][:-1] - data["timestamp"][init_val] * np.ones(len(data["timestamp"][:-1])))
        linesfz.set_ydata(y3[:-1])

    ax.set_ylim(init_force_slider.val, end_force_slider.val)
    fig.canvas.draw_idle()


path = "../data/sensor_testing/paper_analysis/second_attempt/"

res = os.listdir(path)
i = 0

for file in res:
    print(f'[{i}]:' + file)
    i += 1

idx = input("Select idx from test json: ")

f = open(path + res[int(idx)])

data = json.load(f)

f.close()

fig, ax = plt.subplots()
plt.grid()
if args["variables"][0] == "f":
    if args["smooth"] > 0:
        y1 = mean_values(data["fx"], args["smooth"])
        y2 = mean_values(data["fy"], args["smooth"])
        y3 = mean_values(data["fz"], args["smooth"])
    else:
        y1 = data["fx"]
        y2 = data["fy"]
        y3 = data["fz"]

    ax.set_ylabel("Force [N]")
else:
    if args["smooth"] > 0:
        y1 = mean_values(data["mx"], args["smooth"])
        y2 = mean_values(data["my"], args["smooth"])
        y3 = mean_values(data["mz"], args["smooth"])
    else:
        y1 = data["mx"]
        y2 = data["my"]
        y3 = data["mz"]
    ax.set_ylabel("Torque [N/m]")

f_min = np.min([y1, y2, y3])
f_max = np.max([y1, y2, y3])

# Create the figure and the line that we will manipulate

if axis[0]:
    linesfx, = plt.plot(data["timestamp"], y1, '-r')
    ax.legend([linesfx], ['Axis - X'])
if axis[1]:
    linesfy, = plt.plot(data["timestamp"], y2, '-g')
    ax.legend([linesfy], ['Axis - Y'])
if axis[2]:
    linesfz, = plt.plot(data["timestamp"], y3, '-b')
    ax.legend([linesfz], ['Axis - Z'])

if axis[0] and axis[1] and axis[2]:
    ax.legend([linesfx, linesfy, linesfz], ['Fx', 'Fy', 'Fz'])

ax.set_xlabel('Time [s]')
fig.canvas.set_window_title(res[int(idx)])
# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axinit = plt.axes([0.25, 0.1, 0.65, 0.03])
axend = plt.axes([0.25, 0.05, 0.65, 0.03])
init_slider = Slider(
    ax=axinit,
    label='initial time',
    valmin=0.1,
    valmax=data["timestamp"][-1],
    valinit=0,
)
end_slider = Slider(
    ax=axend,
    label='final time',
    valmin=0.1,
    valmax=data["timestamp"][-1],
    valinit=data["timestamp"][-1],
)
axinit = plt.axes([0.1, 0.25, 0.0225, 0.63])
axend = plt.axes([0.05, 0.25, 0.0225, 0.63])
init_force_slider = Slider(
    ax=axinit,
    label='initial force',
    valmin=f_min*1.1,
    valmax=f_max*1.1,
    valinit=f_min*1.1,
    orientation="vertical",
)
end_force_slider = Slider(
    ax=axend,
    label='final force',
    valmin=f_min*1.1,
    valmax=f_max*1.1,
    valinit=f_max*1.1,
    orientation="vertical",
)

init_slider.on_changed(update)
end_slider.on_changed(update)
init_force_slider.on_changed(update)
end_force_slider.on_changed(update)

#
# showx = True
# def buttonx(val):
#     global showx
#     global linesfx
#     showx = not showx
#     print(showx)
#     if showx:
#         linesfx, = plt.plot(data["timestamp"], data["fx"], '-r')
#     else:
#         linesfx.remove()
#
#     plt.plot()
#
#
# axx = plt.axes([0.7, 0.05, 0.1, 0.075])
# bx = Button(axx, "X-Axis")
# bx.on_clicked(buttonx)

plt.show()
