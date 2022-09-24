#!/usr/bin/env python3
import json
import math
import time

import numpy as np
import rospy
from matplotlib import pyplot as plt
from std_msgs.msg import Float64, String, Bool

from config.definitions import ROOT_DIR


class TriggerVisualizer:
    def __init__(self):

        self.data_array = np.empty((1, 13))
        self.first_timestamp = None

        self.x_value = []
        self.y_value = []

        self.classification = None
        self.force_detection = None
        self.calibration = 0.0

        self.is_graph_outdated = False

        f = open(ROOT_DIR + '/data_storage/config/training_config.json')
        self.config = json.load(f)
        f.close()

        self.threshold_start = self.config["force_threshold_start"]
        self.threshold_end = self.config["force_threshold_end"]

        rospy.Subscriber("trigger_data", Float64, self.forces_callback)
        rospy.Subscriber("classification", String, self.class_callback)
        rospy.Subscriber("force_detection", Bool, self.force_detection_callback)
        rospy.Subscriber("calibration", Float64, self.calibration_callback)

        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1)
        self.lines = []
        self.graph_layout(0)
        plt.show()

        while True:
            if self.is_graph_outdated:
                self.update_graph()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            time.sleep(0.001)

    def calibration_callback(self, data):
        self.calibration = data.data

    def forces_callback(self, data):

        rostime = str(rospy.Time.now())
        rostime_in_float = float(str(rostime[:10]) + "." + str(rostime[10:]))

        if self.first_timestamp is None:
            self.first_timestamp = rostime_in_float
            timestamp = 0.0
        else:
            timestamp = rostime_in_float - self.first_timestamp

        self.x_value.append(timestamp)
        self.y_value.append(data.data)

        self.is_graph_outdated = True

    def class_callback(self, data):
        print(data.data)
        self.classification = data.data

    def force_detection_callback(self, data):
        print(data.data)
        self.force_detection = data.data

    def update_graph(self):

        time_vector = self.x_value
        trigger_vector = self.y_value

        last_timestamp = self.x_value[-1]
        self.graph_layout(last_timestamp)

        line = self.ax.plot(time_vector, trigger_vector, "-r")

        self.ax.axhline(y=self.calibration + self.threshold_end, color='y', linestyle='-')
        self.ax.axhline(y=self.calibration - self.threshold_end, color='y', linestyle='-')

        self.ax.axhline(y=self.calibration + self.threshold_start, color='k', linestyle='-')
        self.ax.axhline(y=self.calibration - self.threshold_start, color='k', linestyle='-')

        self.is_graph_outdated = False

    def graph_layout(self, timestamp):
        # for ax in self.ax:
        #     ax.cla()
        #     ax.grid()

        self.ax.cla()
        self.ax.grid()

        time_window = 5
        start_time = timestamp - time_window

        if start_time < 0:
            start_time = 0

        if timestamp < time_window:
            timestamp = time_window

        self.ax.set_title("Joints efforts")
        # self.ax.legend(self.lines[0], ["trigger"])
        self.ax.set_ylim((-2, 2))
        self.ax.set_xlim((start_time, timestamp + 2))

        if self.classification == "Calibrating":
            self.fig.suptitle("CALIBRATING SENSORS", fontsize=40, color='blue')
        else:
            if self.force_detection:
                self.fig.suptitle("FORCE DETECTED", fontsize=40, color="green")
            else:
                self.fig.suptitle("NO FORCE DETECTED", fontsize=40, color='red')


if __name__ == '__main__':
    rospy.init_node("trigger_intreperter", anonymous=True)

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    trigger_visualizer = TriggerVisualizer()
