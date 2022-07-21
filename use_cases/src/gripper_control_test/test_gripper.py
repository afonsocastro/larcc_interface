#!/usr/bin/env python3
import time

import rospy

from lib.src.ArmGripperComm import ArmGripperComm

from gripper.src.RobotiqHand import RobotiqHand

HOST = "192.168.56.2"
PORT = 54321

# --------------------------------------------------------------------
# -------------------------Initialization-----------------------------
# --------------------------------------------------------------------


rospy.init_node("gripper_test", anonymous=True)

arm_gripper_comm = ArmGripperComm()

time.sleep(0.2) # Waiting time to ros nodes properly initiate

arm_gripper_comm.gripper_connect()

if not arm_gripper_comm.state_dic["object_detected"]:
    arm_gripper_comm.gripper_init()

arm_gripper_comm.gripper_disconnect()
