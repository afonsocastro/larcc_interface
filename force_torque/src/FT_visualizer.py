#!/usr/bin/env python3

import time
import numpy as np
import rospy
from matplotlib import pyplot as plt
from geometry_msgs.msg import Wrench, Pose, WrenchStamped


class ActionVisualizer:
    def __init__(self):

        # self.data_array = np.empty((1, 13))
        self.first_timestamp = None

        self.classification = "None"

        self.data_array = []

        self.is_graph_outdated = False

        rospy.Subscriber("wrench", WrenchStamped, self.forces_calback)

        plt.ion()
        self.ax, self.fig  = plt.subplots(1, 1)
        self.lines = []
        self.graph_layout(0)
        plt.show()

        while True:
            if self.is_graph_outdated:
                self.update_graph()

            self.ax.canvas.draw()
            self.ax.canvas.flush_events()
            # time.sleep(0.001)

    def forces_calback(self, data):

        rostime = str(rospy.Time.now())
        rostime_in_float = float(str(rostime[:10]) + "." + str(rostime[10:]))

        if self.first_timestamp is None:
            self.first_timestamp = rostime_in_float
            timestamp = 0.0
        else:
            timestamp = rostime_in_float - self.first_timestamp

        new_vector = np.array(
            [timestamp, data.wrench.force.x, data.wrench.force.y, data.wrench.force.z, data.wrench.torque.x,
             data.wrench.torque.y, data.wrench.torque.z])

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
        graph_color = ["#009cf9", "#ee5f30", "#ffbf4d", "#8900f1", "#82b574", "#00d7ff"]
        # self.lines = []

        last_timestamp = data_array[:, 0][-1]
        self.graph_layout(last_timestamp)

        for i in range(0, 3):
            line = self.fig.plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        for i in range(3, 6):
            line = self.fig.plot(data_array[:, 0], data_array[:, i + 1], graph_color[i])
            self.lines.append(line[0])

        self.is_graph_outdated = False

    def graph_layout(self, timestamp):

        self.fig.cla()
        self.fig.grid()

        time_window = 5
        start_time = timestamp - time_window

        if start_time < 0:
            start_time = 0

        if timestamp < time_window:
            timestamp = time_window

        self.fig.set_title("Built-In FT wrench sensor")
        self.fig.legend(self.lines[0:6], ["Fx", "Fy", "Fz", "Mx", "My", "Mz"])
        self.fig.set_ylim((-100, 100))
        self.fig.set_xlim((start_time, timestamp + 2))


if __name__ == '__main__':
    rospy.init_node("FT_viz_node", anonymous=True)

    time.sleep(0.2) # Waiting time to ros nodes properly initiate

    action_visualizer = ActionVisualizer()




