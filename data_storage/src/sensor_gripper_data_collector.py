import json
import os
import time
import rospy
from larcc_classes.data_storage.DataForLearning import DataForLearning
from matplotlib import pyplot as plt


def add_to_dic(data, dic, first_timestamp):

    if first_time_stamp is None:
        first_timestamp = data.timestamp()
        timestamp = 0.0
    else:
        timestamp = data.timestamp() - first_timestamp

    dic["timestamp"].append(timestamp)
    dic["wrist3"].append(data.joints_position[5])
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
    dic["my"].append(data.wrench_force_torque.torque.y)
    dic["mz"].append(data.wrench_force_torque.torque.z)

    return dic, first_timestamp


if __name__ == "__main__":
    plt.ion()

    rospy.init_node("test_aquisition", anonymous=True)

    rate = rospy.Rate(100)

    data_for_learning = DataForLearning()

    time.sleep(0.2)

    figure, ax = plt.subplots(2)
    linesfx, = ax[0].plot([], [], '-r')
    linesfy, = ax[0].plot([], [], '-g')
    linesfz, = ax[0].plot([], [], '-b')
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(-20, 20)

    linesmx, = ax[1].plot([], [], '-r')
    linesmy, = ax[1].plot([], [], '-g')
    linesmz, = ax[1].plot([], [], '-b')
    ax[1].set_xlim(0, 100)
    ax[1].set_ylim(-2, 2)

    data_dic = {"timestamp": [],
                "wrist3": [],
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
        print(data_for_learning)
        data_dic, first_time_stamp = add_to_dic(data_for_learning, data_dic, first_time_stamp)

        linesfx.set_xdata(data_dic["timestamp"])
        linesfx.set_ydata(data_dic["fx"])
        linesfy.set_xdata(data_dic["timestamp"])
        linesfy.set_ydata(data_dic["fy"])
        linesfz.set_xdata(data_dic["timestamp"])
        linesfz.set_ydata(data_dic["fz"])

        ax[0].relim()
        ax[0].autoscale_view()

        linesmx.set_xdata(data_dic["timestamp"])
        linesmx.set_ydata(data_dic["mx"])
        linesmy.set_xdata(data_dic["timestamp"])
        linesmy.set_ydata(data_dic["my"])
        linesmz.set_xdata(data_dic["timestamp"])
        linesmz.set_ydata(data_dic["mz"])

        ax[1].relim()
        ax[1].autoscale_view()

        figure.canvas.draw()
        figure.canvas.flush_events()

        rate.sleep()

    res = os.listdir('../data/sensor_testing/paper_analysis/second_attempt')

    for file in res:
        print(file)

    name = input("Input file name:\n")

    with open(f'../data//sensor_testing/paper_analysis/second_attempt/{name}.json', 'w') as fp:
        json.dump(data_dic, fp)
