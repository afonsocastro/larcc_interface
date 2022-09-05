import time
import rospy
from data_storage.src.data_aquisition_node import DataForLearning
import numpy as np
from matplotlib import pyplot as plt

def add_to_dic(data, dic, first_timestamp):

    if first_time_stamp is None:
        first_timestamp = data.timestamp()
        timestamp = 0.0
    else:
        timestamp = data.timestamp() - first_timestamp

    dic["timestamp"].append(timestamp)
    dic["j0"].append(data.joints_effort[0])
    dic["j1"].append(data.joints_effort[1])
    dic["j2"].append(data.joints_effort[2])
    dic["j3"].append(data.joints_effort[3])
    dic["j4"].append(data.joints_effort[4])
    dic["j5"].append(data.joints_effort[5])
    dic["fx"].append(data.wrench_force_torque.force.x)
    dic["fy"].append(data.wrench_force_torque.force.y)
    dic["fz"].append(data.wrench_force_torque.force.z)
    dic["mx"].append(data.wrench_force_torque.torque.x)
    dic["mx"].append(data.wrench_force_torque.torque.y)
    dic["mz"].append(data.wrench_force_torque.torque.z)

    return dic, first_timestamp


if __name__ == "__main__":
    plt.ion()

    rospy.init_node("test_aquisition", anonymous=True)

    rate = rospy.Rate(100)

    data_for_learning = DataForLearning()

    time.sleep(0.2)

    figure, ax = plt.subplots()
    lines, = ax.plot([], [], '-')
    ax.set_xlim(0, 60)
    ax.set_ylim(-3, 3)

    data_dic = {"timestamp": [],
                "j0": [],
                "j1": [],
                "j2": [],
                "j3": [],
                "j4": [],
                "j5": [],
                "fx": [],
                "fy": [],
                "fz": [],
                "mx": [],
                "my": [],
                "mz": []}

    first_time_stamp = None
    while not rospy.is_shutdown():

        data_dic, first_time_stamp = add_to_dic(data_for_learning, data_dic, first_time_stamp)

        lines.set_xdata(data_dic["timestamp"])
        lines.set_ydata(data_dic["fz"])

        ax.relim()
        ax.autoscale_view()

        figure.canvas.draw()
        figure.canvas.flush_events()

        rate.sleep()
