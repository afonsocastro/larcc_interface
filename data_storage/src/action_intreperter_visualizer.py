#!/usr/bin/env python3
import json
import math
import time

import numpy as np
import rospy
from matplotlib import pyplot as plt
from std_msgs.msg import String, Float64MultiArray

from config.definitions import ROOT_DIR


class ActionVisualizer:
    def __init__(self):

        self.data_array = np.empty((1, 13))
        self.first_timestamp = None

        self.classification = "None"

        self.data_array = []

        self.is_graph_outdated = False

        f = open(ROOT_DIR + '/data_storage/config/training_config.json')
        self.config = json.load(f)
        f.close()

        rospy.Subscriber("classification", String, self.class_calback)
        rospy.Subscriber("learning_data", Float64MultiArray, self.forces_calback)

        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1)
        self.lines = []
        self.graph_layout(0)
        plt.show()

        while True:
            if self.is_graph_outdated:
                self.update_graph()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.001)

    def forces_calback(self, data):

        new_vector = np.array(data.data)

        rostime = str(rospy.Time.now())
        rostime_in_float = float(str(rostime[:10]) + "." + str(rostime[10:]))

        if self.first_timestamp is None:
            self.first_timestamp = rostime_in_float
            timestamp = 0.0
        else:
            timestamp = rostime_in_float - self.first_timestamp

        new_vector[0] = timestamp

        if len(self.data_array) == 0:
            self.data_array = np.array([new_vector])
        else:
            self.data_array = np.append(self.data_array, [new_vector], axis=0)

        self.is_graph_outdated = True

    def class_calback(self, data):

        print(data.data)
        self.classification = data.data
        self.is_graph_outdated = True

    def update_graph(self):

        data_array = self.data_array
        graph_color = ["-r", "-g", "-b", "-y", "-k", "-m", "-r", "-g", "-b", "-r", "-g", "-b"]
        # self.lines = []

        last_timestamp = data_array[:, 0][-1]
        self.graph_layout(last_timestamp)

        for i in range(0, 6):
            line = self.ax[0].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        for i in range(6, 9):
            line = self.ax[1].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        for i in range(9, 12):
            line = self.ax[2].plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        self.is_graph_outdated = False

    def graph_layout(self, timestamp):
        for ax in self.ax:
            ax.cla()
            ax.grid()

        time_window = 5
        start_time = timestamp - time_window

        if start_time < 0:
            start_time = 0

        if timestamp < time_window:
            timestamp = time_window

        self.ax[0].set_title("Joints efforts")
        self.ax[0].legend(self.lines[0:6], ["J0", "J1", "J2", "J3", "J4", "J5"])
        self.ax[0].set_ylim((-10, 10))
        self.ax[0].set_xlim((start_time, timestamp + 2))

        self.ax[1].set_title("Gripper Forces")
        self.ax[1].legend(self.lines[6:9], ["Fx", "Fy", "Fz"])
        self.ax[1].set_ylim((-100, 100))
        self.ax[1].set_xlim((start_time, timestamp + 2))

        self.ax[2].set_title("Gripper Moments")
        self.ax[2].legend(self.lines[9:12], ["Mx", "My", "Mz"])
        self.ax[2].set_ylim((-10, 10))
        self.ax[2].set_xlim((start_time, timestamp + 2))

        if self.classification == "Calibrating":
            self.fig.suptitle("CALIBRATING SENSORS", fontsize=40, color='red')
        else:
            self.fig.suptitle("Predicted Action: " + self.classification, fontsize=40)


if __name__ == '__main__':
    rospy.init_node("action_intreperter", anonymous=True)

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    action_visualizer = ActionVisualizer()





