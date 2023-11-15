#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
import numpy as np
from colorama import Fore
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
import math

from config.definitions import ROOT_DIR


class DataVisualizer:
    # def __init__(self, file=ROOT_DIR + "/data_storage/data1/global_normalized_train_data_20ms.npy"):
    # def __init__(self, file=ROOT_DIR + "/data_storage/data1/raw_learning_data.npy"):
    # def __init__(self, file=ROOT_DIR + "/data_storage/data2/x_test_global_normalized_data.npy"):
    # def __init__(self, file=ROOT_DIR + "/data_storage/data2/x_test_raw_data.npy"):
    def __init__(self, file=ROOT_DIR + "/data_storage/data3/raw_learning_data.npy"):

        self.dataset = 3

        if self.dataset == 2:
            self.measurements = 20
        elif self.dataset == 1:
            self.measurements = 50
        else:
            self.measurements = 0


        self.idx = 0

        data_matrix = np.load(file)

        # self.data = data_matrix[:, :-1]
        self.data = data_matrix

        self.is_graph_outdated = True

        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1)
        self.lines = []
        plt.show()
        self.update_graph()

        axprev1 = plt.axes([0.125, 0.025, 0.1, 0.05])
        axnext1 = plt.axes([0.8, 0.025, 0.1, 0.05])
        axprev10 = plt.axes([0.23, 0.025, 0.1, 0.05])
        axnext10 = plt.axes([0.695, 0.025, 0.1, 0.05])
        axprev100 = plt.axes([0.335, 0.025, 0.1, 0.05])
        axnext100 = plt.axes([0.59, 0.025, 0.1, 0.05])
        axrandom = plt.axes([0.44, 0.025, 0.145, 0.05])

        bnext1 = Button(axnext1, ' +1')
        bnext1.on_clicked(self.next1)

        bprev1 = Button(axprev1, ' -1')
        bprev1.on_clicked(self.prev1)

        bnext10 = Button(axnext10, ' +10')
        bnext10.on_clicked(self.next10)

        bprev10 = Button(axprev10, ' -10')
        bprev10.on_clicked(self.prev10)

        bnext100 = Button(axnext100, ' +100')
        bnext100.on_clicked(self.next100)

        bprev100 = Button(axprev100, ' -100')
        bprev100.on_clicked(self.prev100)

        brandom = Button(axrandom, ' random')
        brandom.on_clicked(self.random_idx)

        while True:
            if self.is_graph_outdated:
                if self.idx >= self.data.shape[0]:
                    self.idx = self.data.shape[0] - 1

                if self.idx <= 0:
                    self.idx = 0

                self.update_graph()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.01)

    def next1(self, event):
        self.idx += 1
        self.is_graph_outdated = True

    def prev1(self, event):
        self.idx -= 1
        self.is_graph_outdated = True

    def next10(self, event):
        self.idx += 10
        self.is_graph_outdated = True

    def prev10(self, event):
        self.idx -= 10
        self.is_graph_outdated = True

    def next100(self, event):
        self.idx += 100
        self.is_graph_outdated = True

    def prev100(self, event):
        self.idx -= 100
        self.is_graph_outdated = True

    def random_idx(self, event):
        self.idx = random.randint(0, self.data.shape[0])
        self.is_graph_outdated = True

    def update_graph(self):

        for ax in self.ax:
            ax.cla()
            ax.grid()

        data_vector = self.data[self.idx, :, :]

        if self.dataset == 1:
            data_array = np.reshape(data_vector, (self.measurements, int(len(data_vector) / self.measurements)))
        elif self.dataset == 2:
            for t in range(50, 100):
                data_vector[t, 0] += + 0.5
            data_array = data_vector
        elif self.dataset == 3:
            data_array = data_vector[:, :-1]

        graph_color = ["-r", "-g", "-b", "-y", "-k", "-m", "-r", "-g", "-b", "-r", "-g", "-b"]

        joints_max = []
        forces_max = []
        torques_max = []

        for i in range(0, 6):
            joints_max.append(max(abs(data_array[:, i + 1])))
            line = self.ax[0].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        joints_y_lim = math.ceil(1.1 * max(joints_max))

        self.ax[0].set_title("Joints efforts")
        self.ax[0].legend(self.lines[0:6], ["J0", "J1", "J2", "J3", "J4", "J5"])
        self.ax[0].set_ylim((-joints_y_lim, joints_y_lim))

        for i in range(6, 9):
            forces_max.append(max(abs(data_array[:, i + 1])))
            line = self.ax[1].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        forces_y_lim = math.ceil(1.1 * max(forces_max))

        self.ax[1].set_title("Gripper Forces")
        self.ax[1].legend(self.lines[6:9], ["Fx", "Fy", "Fz"])
        self.ax[1].set_ylim((-forces_y_lim, forces_y_lim))

        for i in range(9, 12):
            torques_max.append(max(abs(data_array[:, i + 1])))
            line = self.ax[2].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        torques_y_lim = math.ceil(1.1 * max(torques_max))

        self.ax[2].set_title("Gripper Moments")
        self.ax[2].legend(self.lines[9:12], ["Mx", "My", "Mz"])
        self.ax[2].set_ylim((-torques_y_lim, torques_y_lim))

        self.fig.suptitle(str(self.idx), fontsize=20)

        self.is_graph_outdated = False


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for visualizer script")
    # parser.add_argument("-p", "--path", type=str, default="/data_storage/data/raw_learning_data/",
    #                     help="The relative path to the .npy file")

    parser.add_argument("-p", "--path", type=str, default="predicted_learning_data",
                        help="The relative path to the .npy file")

    parser.add_argument("-f", "--file", type=str, default="raw_learning_data.npy",
                        help="The relative path to the .npy file")

    parser.add_argument("-c", "--config_file", type=str, default="data_storage_config",
                        help="If argmument is present, activates gripper")

    # parser.add_argument("-c", "--config_file", type=str, default="training_config",
    #                     help="If argmument is present, activates gripper")

    args = vars(parser.parse_args())

    data_vis = DataVisualizer()


