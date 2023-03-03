#!/usr/bin/env python3
import argparse
import json
import os
import sys

from config.definitions import ROOT_DIR
from larcc_classes.ur10e_control.ArmGripperComm import ArmGripperComm
import rospy
import time


parser = argparse.ArgumentParser(description="Arguments for trainning script")
parser.add_argument("-m", "--movement", type=str, default="",
                    help="It is the name of the movement configuration JSON in the config directory of this package")
parser.add_argument("-pl", "--position_list", type=str, default="positions",
                    help="It is the name of the configuration JSON containing the list of positions in the config directory of this package")

args = vars(parser.parse_args())

path = ROOT_DIR + "/use_cases/config/"

try:
    f = open(path + args['position_list'] + ".json")
    positions = json.load(f)
    positions = positions["positions"]
    f.close()

except:
    rospy.logerr("Invalid positions file! Closing...")
    sys.exit(0)

rospy.init_node("arm_gripper_movement", anonymous=True)

arm_gripper_comm = ArmGripperComm()

time.sleep(0.2)

arm_gripper_comm.gripper_connect()

#arm_gripper_comm.gripper_status()

if not arm_gripper_comm.state_dic["activation_completed"]: 
    arm_gripper_comm.gripper_init()

if args['movement'] == "":
    res = os.listdir(path)
    res.remove(args['position_list'] + ".json")

    while True:
        i = 0

        for file in res:
            print(f'[{i}]:' + file)
            i += 1

        idx = input("Select idx from test json: ")

        try:
            f = open(path + res[int(idx)])
            config = json.load(f)
            f.close()
        
        except:
            rospy.logerr("Invalid file! Closing...")
            arm_gripper_comm.gripper_disconnect()
            sys.exit(0)

        for pos, gripper in config["positions"]:
            pos = positions[pos]
            arm_gripper_comm.move_arm_to_joints_state(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

            if gripper == 1:
                arm_gripper_comm.gripper_open_fast()
            if gripper == -1:
                arm_gripper_comm.gripper_close_fast()

else:
    try:
        f = open(path + args["movement"] + '.json')
        config = json.load(f)
        f.close()
    
    except:
        rospy.logerr("Invalid file! Closing...")
        arm_gripper_comm.gripper_disconnect()
        sys.exit(0)

    for pos, gripper in config["positions"]:
        pos = positions[pos]
        arm_gripper_comm.move_arm_to_joints_state(pos[0], pos[1], pos[2], pos[3], pos[4], pos[5])

        if gripper == 1:
            arm_gripper_comm.gripper_open_fast()
        if gripper == -1:
            arm_gripper_comm.gripper_close_fast()

arm_gripper_comm.gripper_disconnect()
