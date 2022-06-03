#!/usr/bin/env python3

import rospy
from gripper.msg import ForceTorque
from sensor_msgs.msg import JointState

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import collections


class ForceTorqueValues:
    def __init__(self):
        self.wrench_force = []
        self.wrench_torque = []
        self.joints_effort = []
        self.gripper_current = 0


force_torque_values = ForceTorqueValues()


def callback(data):
    force_torque_values.wrench_force = data.WrenchForce
    force_torque_values.wrench_torque = data.WrenchTorque
    force_torque_values.joints_effort = data.JointsEffort.data
    force_torque_values.gripper_current = data.GripperCurrent.data


# function to update the data
def my_function(i):

    ax_wf.cla()
    for x in range(1, 4):
        wf["wf{0}".format(x)].popleft()

        if x == 1:
            wf["wf{0}".format(x)].append(force_torque_values.wrench_force.x)
            legend = "X Axis"
        elif x == 2:
            wf["wf{0}".format(x)].append(force_torque_values.wrench_force.y)
            legend = "Y Axis"
        elif x == 3:
            wf["wf{0}".format(x)].append(force_torque_values.wrench_force.z)
            legend = "Z Axis"

        ax_wf.plot(wf["wf{0}".format(x)], label=legend)
        ax_wf.scatter(len(wf["wf{0}".format(x)]) - 1, wf["wf{0}".format(x)][-1])
        ax_wf.text(len(wf["wf{0}".format(x)]) - 1, wf["wf{0}".format(x)][-1] + 0.1,
                   '%.3f' % (wf["wf{0}".format(x)][-1]))
        ax_wf.set_ylim(-20, 20)  # wrench force
        ax_wf.set_ylabel('N')
        ax_wf.set_title('Wrench Force')
        ax_wf.legend(loc='upper left', frameon=False)
        ax_wf.axes.get_xaxis().set_visible(False)

    ax_wt.cla()
    for x in range(1, 4):
        wt["wt{0}".format(x)].popleft()

        if x == 1:
            wt["wt{0}".format(x)].append(force_torque_values.wrench_torque.x)
            legend = "X Axis"
        elif x == 2:
            wt["wt{0}".format(x)].append(force_torque_values.wrench_torque.y)
            legend = "Y Axis"
        elif x == 3:
            wt["wt{0}".format(x)].append(force_torque_values.wrench_torque.z)
            legend = "Z Axis"

        ax_wt.plot(wt["wt{0}".format(x)], label=legend)
        ax_wt.scatter(len(wt["wt{0}".format(x)]) - 1, wt["wt{0}".format(x)][-1])
        ax_wt.text(len(wt["wt{0}".format(x)]) - 1, wt["wt{0}".format(x)][-1] + 0.1,
                   '%.3f' % (wt["wt{0}".format(x)][-1]))
        ax_wt.set_ylim(-8, 8)  # wrench force
        ax_wt.set_ylabel('Nm')
        ax_wt.set_title('Wrench Torque')
        ax_wt.legend(loc='upper left', frameon=False)
        ax_wt.axes.get_xaxis().set_visible(False)


    ax_je.cla()
    for x in range(1, 7):
        je["je{0}".format(x)].popleft()
        je["je{0}".format(x)].append(force_torque_values.joints_effort[x - 1])

        ax_je.plot(je["je{0}".format(x)], label="Joint {0}".format(x))
        ax_je.scatter(len(je["je{0}".format(x)]) - 1, je["je{0}".format(x)][-1])
        ax_je.text(len(je["je{0}".format(x)]) - 1, je["je{0}".format(x)][-1] + 0.1, '%.3f' % (je["je{0}".format(x)][-1]))
        ax_je.set_ylim(-5, 3)  # joints effort
        ax_je.set_ylabel('Nm')
        ax_je.set_title('Joints Effort')
        ax_je.legend(loc='upper left', frameon=False)
        ax_je.axes.get_xaxis().set_visible(False)

    gc.popleft()
    gc.append(force_torque_values.gripper_current)  # clear axis
    ax_gc.cla()  # plot gripper current
    ax_gc.plot(gc)
    ax_gc.scatter(len(gc) - 1, gc[-1])
    ax_gc.text(len(gc) - 1, gc[-1] + 2, gc[-1])
    ax_gc.set_ylabel('mA')
    ax_gc.set_title('Gripper Current')
    ax_gc.set_ylim(-1, 100)  # start collections with zeros
    ax_gc.axes.get_xaxis().set_visible(False)


if __name__ == '__main__':

    rospy.init_node('graphs_node', anonymous=True)
    rate = rospy.Rate(10.0)  # 10hz

    rospy.Subscriber("force_torque_values", ForceTorque, callback)

    wf = {}
    for x in range(1, 4):
        wf["wf{0}".format(x)] = collections.deque(np.zeros(10))

    wt = {}
    for x in range(1, 4):
        wt["wt{0}".format(x)] = collections.deque(np.zeros(10))

    je = {}
    for x in range(1, 7):
        je["je{0}".format(x)] = collections.deque(np.zeros(10))

    gc = collections.deque(np.zeros(10))  # define and adjust figure
    fig = plt.figure(figsize=(24, 48), facecolor='#DEDEDE')
    ax_wf = plt.subplot(221)
    ax_wt = plt.subplot(222)
    ax_je = plt.subplot(223)
    ax_gc = plt.subplot(224)

    ax_wf.set_facecolor('#DEDEDE')
    ax_wt.set_facecolor('#DEDEDE')
    ax_je.set_facecolor('#DEDEDE')
    ax_gc.set_facecolor('#DEDEDE')

    ani = FuncAnimation(fig, my_function, interval=1)
    plt.show()

    rospy.spin()
