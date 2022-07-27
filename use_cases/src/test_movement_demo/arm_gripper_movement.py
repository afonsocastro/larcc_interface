#!/usr/bin/env python3
import argparse
import json
import os

import tf2_ros
from lib.src.ArmGripperComm import ArmGripperComm
import rospy
import time

parser = argparse.ArgumentParser(description="Arguments for trainning script")
parser.add_argument("-m", "--movement", type=str, default="",
                    help="It is the name of the movement configuration JSON in the config directory of this package")

args = vars(parser.parse_args())

path = "../../config/"

if args['movement'] == "":
    res = os.listdir(path)
    i = 0

    for file in res:
        print(f'[{i}]:' + file)
        i += 1

    idx = input("Select idx from test json: ")

    f = open(path + res[int(idx)])
    config = json.load(f)
    f.close()
else:
    f = open(path + args["movement"] + '.json')
    config = json.load(f)
    f.close()

rospy.init_node("arm_gripper_movement", anonymous=True)

arm_gripper_comm = ArmGripperComm()

time.sleep(0.2)

arm_gripper_comm.gripper_connect()

for pos in config["positions"]:
    arm_gripper_comm.move_arm_to_joints_state(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

    if pos[6] == 1:
        arm_gripper_comm.gripper_open_fast()
    if pos[6] == -1:
        arm_gripper_comm.gripper_close_fast()

arm_gripper_comm.gripper_disconnect()
