#!/usr/bin/env python3
import json

import numpy as np
import rospy
import time
from data_storage.src.data_aquisition_node import DataForLearning
from matplotlib import pyplot as plt


def calc_data_mean(data):
    new_values = np.array([data.wrench_force_torque.force.z/10, data.wrench_force_torque.torque.x,
                           data.wrench_force_torque.torque.y, data.wrench_force_torque.torque.z])

    return np.mean(abs(new_values))


if __name__ == '__main__':

    # ---------------------------------------------------------------------------------------------
    # --------------------------------------INPUT VARIABLES----------------------------------------
    # ---------------------------------------------------------------------------------------------

    f = open('../../config/data_storage_config.json')

    config = json.load(f)

    f.close()

    rospy.init_node("trigger_visualizer", anonymous=True)

    rate = rospy.Rate(config["rate"])

    time.sleep(0.2)

    data_for_learning = DataForLearning()
    list_gripper_calibration = []
    for i in range(0, 20):
        list_gripper_calibration.append(calc_data_mean(data_for_learning))
        rate.sleep()

    calibration = np.mean(list_gripper_calibration)

    ulim = calibration + config["force_threshold_start"]
    dlim = calibration - config["force_threshold_start"]

    mean_vector = []
    time_vector = []

    plt.show()

    axes = plt.gca()
    axes.set_xlim(-0.1, 10)
    axes.set_ylim(dlim - 4 * config["force_threshold_start"], ulim + 4 * config["force_threshold_start"])
    line, = axes.plot([0, 100], [dlim, dlim], 'r-')
    uplim = axes.plot([0, 100], [ulim, ulim], 'b-')
    dplim = axes.plot([0, 100], [dlim, dlim], 'b-')

    cal = axes.plot([0, 100], [dlim, dlim], 'k-')

    axes.title.set_title(str(round(cal, 5)) + "NO FORCE")

    st = time.time()
    while not rospy.is_shutdown():
        values = np.array([data_for_learning.wrench_force_torque.force.z / 10, data_for_learning.wrench_force_torque.torque.x,
                           data_for_learning.wrench_force_torque.torque.y, data_for_learning.wrench_force_torque.torque.z])

        trigger = np.mean(abs(values))

        mean_vector.append(trigger)

        et = time.time()
        time_vector.append(round(et-st, 2))

        line.set_xdata(time_vector)
        line.set_ydata(mean_vector)

        plt.draw()
        plt.pause(1e-17)
        rate.sleep()
