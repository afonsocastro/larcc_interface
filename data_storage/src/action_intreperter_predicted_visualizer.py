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


class PredictedDataVisualizer:
    def __init__(self,  path=ROOT_DIR + "/data_storage/data/predicted_learning_data/", config_file="data_storage_config"):

        self.idx = 0

        res = os.listdir(path)
        i = 0

        for file in res:
            print(f'[{i}]:' + file)
            i += 1

        idx = int(input("Select idx for the matrix: "))
        self.file_name = res[idx]

        f = open(path + res[idx])
        data_json = json.load(f)
        f.close()

        f = open(ROOT_DIR + '/data_storage/config/' + config_file + '.json')
        self.config = json.load(f)
        f.close()

        self.prediction = "None"
        for classification in self.config["action_classes"]:
            if classification in self.file_name:
                self.prediction = classification

        self.measurements = int(self.config["time"] * self.config["rate"])
        self.n_variables = len(self.config["data"])

        self.data = np.array(data_json["data_predicted"])[:, :-1]
        self.classifications = np.array(data_json["data_predicted"])[:, -1]
        self.outputs = data_json["predictions_confidence"]

        self.is_graph_outdated = True

        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1)
        self.lines = []
        plt.show()

        time_max_prev = 0

        self.time = []

        self.fx = []
        self.fy = []
        self.fz = []

        self.mx = []
        self.my = []
        self.mz = []

        self.j0 = []
        self.j1 = []
        self.j2 = []
        self.j3 = []
        self.j4 = []
        self.j5 = []

        self.time_divisions = []

        for i in range(self.data.shape[0]):

            data_vector = self.data[i, :]
            data_array = np.reshape(data_vector, (self.measurements, int(len(data_vector) / self.measurements)))
            print(i)

            new_time = data_array[:, 0] + [time_max_prev]*50

            self.time.extend(new_time)

            time_max_prev = self.time[-1]
            self.time_divisions.append(time_max_prev)

            self.j0.extend(data_array[:, 1])
            self.j1.extend(data_array[:, 2])
            self.j2.extend(data_array[:, 3])
            self.j3.extend(data_array[:, 4])
            self.j4.extend(data_array[:, 5])
            self.j5.extend(data_array[:, 6])

            self.fx.extend(data_array[:, 7])
            self.fy.extend(data_array[:, 8])
            self.fz.extend(data_array[:, 9])

            self.mx.extend(data_array[:, 10])
            self.my.extend(data_array[:, 11])
            self.mz.extend(data_array[:, 12])

        self.joint_values = [self.j0, self.j1, self.j2, self.j3, self.j4, self.j5]
        self.f_values = [self.fx, self.fy, self.fz]
        self.m_values = [self.mx, self.my, self.mz]

        self.update_graph()

        # axprev1 = plt.axes([0.125, 0.025, 0.1, 0.05])
        # axnext1 = plt.axes([0.8, 0.025, 0.1, 0.05])
        # axprev10 = plt.axes([0.23, 0.025, 0.1, 0.05])
        # axnext10 = plt.axes([0.695, 0.025, 0.1, 0.05])
        # axprev100 = plt.axes([0.335, 0.025, 0.1, 0.05])
        # axnext100 = plt.axes([0.59, 0.025, 0.1, 0.05])
        # axrandom = plt.axes([0.44, 0.025, 0.145, 0.05])
        #
        # axprev_inaccurate = plt.axes([0.03, 0.025, 0.09, 0.05])
        # axnext_inaccurate = plt.axes([0.905, 0.025, 0.09, 0.05])
        #
        # bnext1 = Button(axnext1, ' +1')
        # bnext1.on_clicked(self.next1)
        #
        # bprev1 = Button(axprev1, ' -1')
        # bprev1.on_clicked(self.prev1)
        #
        # bnext10 = Button(axnext10, ' +10')
        # bnext10.on_clicked(self.next10)
        #
        # bprev10 = Button(axprev10, ' -10')
        # bprev10.on_clicked(self.prev10)
        #
        # bnext100 = Button(axnext100, ' +100')
        # bnext100.on_clicked(self.next100)
        #
        # bprev100 = Button(axprev100, ' -100')
        # bprev100.on_clicked(self.prev100)
        #
        # brandom = Button(axrandom, ' random')
        # brandom.on_clicked(self.random_idx)
        #
        # bprev_inaccurate = Button(axprev_inaccurate, ' Previous Inaccuracy')
        # bprev_inaccurate.on_clicked(self.prev_inaccurate)
        #
        # bnext_inaccurate = Button(axnext_inaccurate, ' Next Inaccuracy')
        # bnext_inaccurate.on_clicked(self.next_inaccurate)

        while True:
            # if self.is_graph_outdated:
            #     if self.idx >= self.data.shape[0]:
            #         self.idx = self.data.shape[0] - 1
            #
            #     if self.idx <= 0:
            #         self.idx = 0
            #
            #     self.update_graph()

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

    def prev_inaccurate(self, event):
        for i in range(1, 500):
            if self.idx - i <= 0:
                self.idx = 0
                break

            true_classification = str(self.config["action_classes"][int(self.classifications[self.idx - i])])
            if true_classification != self.prediction:
                self.idx -= i
                break

        self.is_graph_outdated = True

    def next_inaccurate(self, event):
        for i in range(1, 500):
            if self.idx + i >= self.data.shape[0]:
                self.idx = self.data.shape[0] - 1
                break

            true_classification = str(self.config["action_classes"][int(self.classifications[self.idx + i])])
            if true_classification != self.prediction:
                self.idx += i
                break

        self.is_graph_outdated = True

    def update_graph(self):

        for ax in self.ax:
            ax.cla()
            ax.grid()

            for division in self.time_divisions:
                ax.axvline(x=division, color='k')

        data_vector = self.data[self.idx, :]

        data_array = np.reshape(data_vector, (self.measurements, int(len(data_vector) / self.measurements)))
        graph_color = ["-r", "-g", "-b", "-y", "-k", "-m", "-r", "-g", "-b", "-r", "-g", "-b"]

        joints_max = []
        forces_max = []
        torques_max = []
        i = 0
        for joints in self.joint_values:
            joints_max.append(max([abs(ele) for ele in joints]))
            line = self.ax[0].plot(self.time, joints, graph_color[i])
            self.lines.append(line[0])
            i += 1

        joints_y_lim = math.ceil(1.1 * max(joints_max))

        self.ax[0].set_title("Joints efforts")
        self.ax[0].legend(self.lines[0:6], ["J1", "J2", "J3", "J4", "J5", "J6"])
        self.ax[0].set_ylim((-joints_y_lim, joints_y_lim))

        for forces in self.f_values:
            forces_max.append(max([abs(ele) for ele in forces]))
            line = self.ax[1].plot(self.time, forces, graph_color[i])
            self.lines.append(line[0])
            i += 1

        forces_y_lim = math.ceil(1.1 * max(forces_max))

        self.ax[1].set_title("Gripper Forces")
        self.ax[1].legend(self.lines[6:9], ["Fx", "Fy", "Fz"])
        self.ax[1].set_ylim((-forces_y_lim, forces_y_lim))

        for torques in self.m_values:
            torques_max.append(max([abs(ele) for ele in torques]))
            line = self.ax[2].plot(self.time, torques, graph_color[i])
            self.lines.append(line[0])
            i += 1

        torques_y_lim = math.ceil(1.1 * max(torques_max))

        self.ax[2].set_title("Gripper Moments")
        self.ax[2].legend(self.lines[9:12], ["Mx", "My", "Mz"])
        self.ax[2].set_ylim((-torques_y_lim, torques_y_lim))

        true_classification = str(self.config["action_classes"][int(self.classifications[self.idx])])

        if true_classification == self.prediction:
            outcome = "Accurate"
        else:
            outcome = "Fail"
        #
        # self.fig.suptitle("Predicted=" + self.prediction + ", True=" +
        #                   true_classification + ", " + outcome +
        #                   ", " + str(self.idx) + " => [" + str(round(self.outputs[self.idx][0], 3)) + ", " +
        #                   str(round(self.outputs[self.idx][1], 3)) + ", " + str(round(self.outputs[self.idx][2], 3))
        #                   + ", " + str(round(self.outputs[self.idx][3], 3)) + "]", fontsize=20)

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

    data_vis = PredictedDataVisualizer(path=ROOT_DIR + "/data_storage/data/predicted_learning_data/",
                              config_file=args["config_file"])

