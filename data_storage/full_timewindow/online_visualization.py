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

        self.cnn_classification = "None"
        self.transformer_classification = "None"

        self.data_array = []

        self.is_graph_outdated = False

        f = open(ROOT_DIR + '/data_storage/config/training_config.json')
        self.config = json.load(f)
        f.close()

        rospy.Subscriber("cnn_classification", String, self.cnn_class_calback)
        rospy.Subscriber("transformer_classification", String, self.transformer_class_calback)
        rospy.Subscriber("learning_data", Float64MultiArray, self.forces_calback)

        plt.ion()
        self.fig, self.ax = plt.subplots(3, 1)
        self.lines = []

        self.cnn_model_name = self.fig.text(0.18, 0.93, "cnn: ", ha="center", fontsize=18)
        self.cnn_output_text = self.fig.text(0.25, 0.93, "", ha="center", fontsize=18)
        self.cnn_percentage_text = self.fig.text(0.34, 0.93, "", ha="center", fontsize=18)

        self.transformer_model_name = self.fig.text(0.69, 0.93, "transformer: ", ha="center", fontsize=18)
        self.transformer_output_text = self.fig.text(0.79, 0.93, "", ha="center", fontsize=18)
        self.transformer_percentage_text = self.fig.text(0.88, 0.93, "", ha="center", fontsize=18)


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

    def cnn_class_calback(self, data):

        print(data.data)
        self.cnn_classification = data.data
        self.is_graph_outdated = True

    def transformer_class_calback(self, data):

        print(data.data)
        self.transformer_classification = data.data
        self.is_graph_outdated = True

    def parse_label(self, s):
        parts = s.strip().split()
        if len(parts) == 2:
            label, percent = parts
            return label, percent
        return parts[0], None

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

        self.cnn_output_text.set_text("")
        self.cnn_percentage_text.set_text("")
        self.cnn_output_text.set_bbox(dict(facecolor="white", alpha=0.5))
        self.cnn_output_text.set_fontsize(18)
        self.cnn_output_text.set_fontweight("normal")

        self.transformer_output_text.set_text("")
        self.transformer_percentage_text.set_text("")
        self.transformer_output_text.set_bbox(dict(facecolor="white", alpha=0.5))
        self.transformer_output_text.set_fontsize(18)
        self.transformer_output_text.set_fontweight("normal")


        if self.cnn_classification == "Calibrating" or self.transformer_classification == "Calibrating":
            self.cnn_model_name.set_text("")
            self.transformer_model_name.set_text("")
            self.fig.suptitle("CALIBRATING SENSORS", fontsize=40, color='red')
        else:
            self.fig.suptitle("Predicted Actions:", fontsize=14)
            self.cnn_model_name.set_text("cnn: ")
            self.transformer_model_name.set_text("transformer: ")

            label_cnn, percent_cnn = self.parse_label(self.cnn_classification)
            label_transformer, percent_transformer = self.parse_label(self.transformer_classification)
            if percent_cnn is None:
                self.cnn_output_text.set_text(label_cnn)
                self.transformer_output_text.set_text(label_cnn)
            else:
                self.cnn_output_text.set_fontsize(36)
                self.cnn_output_text.set_fontweight("bold")
                self.cnn_output_text.set_bbox(dict(facecolor="green", alpha=0.3))
                self.cnn_output_text.set_text(label_cnn.upper())
                self.cnn_percentage_text.set_text("("+percent_cnn+")")

                self.transformer_output_text.set_fontsize(36)
                self.transformer_output_text.set_fontweight("bold")
                self.transformer_output_text.set_bbox(dict(facecolor="green", alpha=0.3))
                self.transformer_output_text.set_text(label_transformer.upper())
                self.transformer_percentage_text.set_text("(" + percent_transformer + ")")


if __name__ == '__main__':
    rospy.init_node("online_visualization", anonymous=True)

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    action_visualizer = ActionVisualizer()





