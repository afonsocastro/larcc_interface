#!/usr/bin/env python3

from lib.src.ArmGripperComm import ArmGripperComm
import rospy
import time

rospy.init_node("demo3", anonymous=True)

arm_gripper_comm = ArmGripperComm()
time.sleep(0.2)

# posisioning in known pose
arm_gripper_comm.gripper_connect()

arm_gripper_comm.move_arm_to_initial_pose()

# Aproach a point near the wooden block
arm_gripper_comm.move_arm_to_joints_state(0.6486111, -1.403100, 1.90815, -2.084144, -1.587375, 0.613297)

# aproach the wooden block
arm_gripper_comm.move_arm_to_joints_state(0.6030416488647461, -1.3459513944438477, 2.0025203863727015,
                                          -2.2264419994749964, -1.5000456015216272, 0.5708098411560059)

# Close gripper
arm_gripper_comm.gripper_open_fast()

# Lift wooden
arm_gripper_comm.move_arm_to_joints_state(0.6486111, -1.403100, 1.90815, -2.084144, -1.587375, 0.613297)

# return to initial known pose
arm_gripper_comm.move_arm_to_initial_pose()


arm_gripper_comm.gripper_disconnect()