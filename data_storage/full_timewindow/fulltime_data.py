#!/usr/bin/env python3
import argparse
from datetime import datetime
import json
import os
import time
from std_msgs.msg import String

from larcc_classes.data_storage.DataForLearning import DataForLearning
import numpy as np
import rospy


class FulltimeData:
    def __init__(self):
        self.vector_data = None
        self.data_for_learning = DataForLearning()

        time.sleep(0.2)

        self.actions = ["PUXAR", "EMPURRAR", "ABANAR", "TORCER"]
        # self.actions = ["PULL", "PUSH", "SHAKE", "TWIST"]

        self.first_time_stamp = None
        self.vector_data = np.empty((0, 14))

        dic_offset_calibration = {"fx": [],
                                  "fy": [],
                                  "fz": [],
                                  "mx": [],
                                  "my": [],
                                  "mz": [],
                                  "j0": [],
                                  "j1": [],
                                  "j2": [],
                                  "j3": [],
                                  "j4": [],
                                  "j5": []}

        print("CALIBRATING...")
        rate = rospy.Rate(100)
        for i in range(0, 50):
            # print(self.data_for_learning)
            dic_offset_calibration["fx"].append(self.data_for_learning.wrench_force_torque.force.x)
            dic_offset_calibration["fy"].append(self.data_for_learning.wrench_force_torque.force.y)
            dic_offset_calibration["fz"].append(self.data_for_learning.wrench_force_torque.force.z)
            dic_offset_calibration["mx"].append(self.data_for_learning.wrench_force_torque.torque.x)
            dic_offset_calibration["my"].append(self.data_for_learning.wrench_force_torque.torque.y)
            dic_offset_calibration["mz"].append(self.data_for_learning.wrench_force_torque.torque.z)

            dic_offset_calibration["j0"].append(self.data_for_learning.joints_effort[0])
            dic_offset_calibration["j1"].append(self.data_for_learning.joints_effort[1])
            dic_offset_calibration["j2"].append(self.data_for_learning.joints_effort[2])
            dic_offset_calibration["j3"].append(self.data_for_learning.joints_effort[3])
            dic_offset_calibration["j4"].append(self.data_for_learning.joints_effort[4])
            dic_offset_calibration["j5"].append(self.data_for_learning.joints_effort[5])
            rate.sleep()

        self.dic_offset = self.offset_calculation(dic_offset_calibration)
        print("CALIBRATED")
        rospy.Subscriber("ground_truth", String, self.callback_interface)

    def callback_interface(self, msg):

        classification = msg.data

        classification_int = None
        if classification.upper() == self.actions[0]:
            classification_int = 0
        elif classification.upper() == self.actions[1]:
            classification_int = 1
        elif classification.upper() == self.actions[2]:
            classification_int = 2
        elif classification.upper() == self.actions[3]:
            classification_int = 3
        elif classification.upper() == 'END':
            save_experiment = input("Save data? (s/n)")
            if "s" in save_experiment.lower():
                self.save_trainnning_data(self.vector_data)
                print("Data saved")
            else:
                print("Data NOT saved")

        if classification_int is not None:
            self.add_to_vector(classification_int)

    def add_to_vector(self, class_int):

        data = self.data_for_learning

        if self.first_time_stamp is None:
            self.first_time_stamp = data.timestamp()
            timestamp = 0.0
        else:
            timestamp = data.timestamp() - self.first_time_stamp

        new_data = np.array([timestamp, data.joints_effort[0] - self.dic_offset["j0"],
                             data.joints_effort[1] - self.dic_offset["j1"],
                             data.joints_effort[2] - self.dic_offset["j2"],
                             data.joints_effort[3] - self.dic_offset["j3"],
                             data.joints_effort[4] - self.dic_offset["j4"],
                             data.joints_effort[5] - self.dic_offset["j5"],
                             data.wrench_force_torque.force.x - self.dic_offset["fx"],
                             data.wrench_force_torque.force.y - self.dic_offset["fy"],
                             data.wrench_force_torque.force.z - self.dic_offset["fz"],
                             data.wrench_force_torque.torque.x - self.dic_offset["mx"],
                             data.wrench_force_torque.torque.y - self.dic_offset["my"],
                             data.wrench_force_torque.torque.z - self.dic_offset["mz"],
                             class_int])

        self.vector_data = np.append(self.vector_data, [new_data], axis=0)

    def offset_calculation(self, dic):

        dic_offset_mean = {}

        for key in dic:
            dic_offset_mean[key] = np.mean(dic[key])

        return dic_offset_mean

    def save_trainnning_data(self, data):

        path = "data"

        files = os.listdir(path)

        file_exists = False

        for file in files:
            if file.find("raw_learning_data.") != -1:
                print("FILE FOUND")
                prev_data_array = np.load(path + "/raw_learning_data.npy")
                data = np.append(prev_data_array, [data], axis=0)
                file_exists = True

        if file_exists is False:
            print("FILE NOT FOUND")
            # data_aux = np.zeros((data.shape[0], data.shape[1], 0))
            # print(data_aux)
            # data_aux[:, :, 0] = data
            np.save(path + "/raw_learning_data.npy", [data])
        else:
            print("SAVING FOUND FILE")
            np.save(path + "/raw_learning_data.npy", data)


if __name__ == '__main__':
    rospy.init_node("training_data_aquisition", anonymous=True)
    ftd = FulltimeData()

    rospy.spin()
