#!/usr/bin/env python3

import os
import time
import numpy as np
import rospy
from std_msgs.msg import String
from larcc_classes.data_storage.DataForLearning import DataForLearning


class FulltimeData:

    def __init__(self):

        self.data_for_learning = DataForLearning()

        self.actions = ["PUXAR", "EMPURRAR", "ABANAR", "TORCER"]

        self.vector_data = np.empty((0, 14))
        # self.current_class = None
        # self.recording = False
        self.first_time_stamp = None

        time.sleep(0.2)

        # ---------------- CALIBRATION ----------------

        dic_offset_calibration = {
            "fx": [], "fy": [], "fz": [],
            "mx": [], "my": [], "mz": [],
            "j0": [], "j1": [], "j2": [],
            "j3": [], "j4": [], "j5": []
        }

        print("CALIBRATING...")

        rate = rospy.Rate(100)

        for _ in range(50):

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

        # ---------------- DATA ACQUISITION LOOP ----------------

        # self.acquire_loop()

    # ------------------------------------------------------------

    def callback_interface(self, msg):

        classification = msg.data.upper()

        # if classification == "START":
        #     print("START RECORDING")
        #     self.vector_data = []
        #     self.first_time_stamp = None
        #     self.recording = True
        #     return

        if classification == "END":

            print("END RECORDING")

            # self.recording = False

            save_experiment = input("Save data? (s/n) ")

            if "s" in save_experiment.lower():
                self.save_trainnning_data(np.array(self.vector_data))
                print("Data saved")
            else:
                print("Data NOT saved")

            return

        if classification in self.actions:
            # self.current_class = self.actions.index(classification)
            self.add_to_vector(self.actions.index(classification))

    # ------------------------------------------------------------

    # def acquire_loop(self):
    #
    #     rate = rospy.Rate(100)
    #
    #     while not rospy.is_shutdown():
    #
    #         if self.recording and self.current_class is not None:
    #             self.add_to_vector(self.current_class)
    #
    #         rate.sleep()

    # ------------------------------------------------------------

    def add_to_vector(self, class_int):

        data = self.data_for_learning

        if self.first_time_stamp is None:
            self.first_time_stamp = data.timestamp()
            timestamp = 0.0
        else:
            timestamp = data.timestamp() - self.first_time_stamp

        new_data = np.array([
            timestamp,
            data.joints_effort[0] - self.dic_offset["j0"],
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
            class_int
        ])

        # self.vector_data.append(new_data)
        self.vector_data = np.append(self.vector_data, [new_data], axis=0)

    # ------------------------------------------------------------

    def offset_calculation(self, dic):

        dic_offset_mean = {}

        for key in dic:
            dic_offset_mean[key] = np.mean(dic[key])

        return dic_offset_mean

    # ------------------------------------------------------------

    def save_trainnning_data(self, data):

        filepath = "data/raw_learning_data.npy"

        if os.path.exists(filepath):
            prev_data = np.load(filepath, allow_pickle=True)
            print("prev_data.shape")
            print(prev_data.shape)
            print("data.shape")
            print(data.shape)
            # new_data = np.concatenate((prev_data, [data]), axis=0)
            new_data = np.append(prev_data, [data], axis=0)
            # np.save(filepath, data)
        else:
            new_data = np.array([data])

        np.save(filepath, new_data)


# ------------------------------------------------------------

if __name__ == '__main__':

    rospy.init_node("training_data_aquisition", anonymous=True)

    FulltimeData()

    rospy.spin()