#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import json
from lib import ArmGripperComm as ag
from time import sleep
import time

print("All imported")

state_dic = {"arm_initial_pose" : None,
             "arm_pose_goal": None,
             "arm_joints_goal": None,
             "gripper_closed": None,
             "gripper_active": None,
             "have_object": None}


def gripper_response_callback(data):
    global state_dic

    state_dic = ag.gripper_response(data.data, state_dic)


def arm_response_callback(data):
    global state_dic

    state_dic = ag.arm_response(data.data, pub_gripper, state_dic)



if __name__ == '__main__':
    rospy.init_node('demo2', anonymous=True)

    rospy.Subscriber("gripper_response", String, gripper_response_callback)
    pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

    rospy.Subscriber("arm_response", String, arm_response_callback)
    pub_arm = rospy.Publisher('arm_request', String, queue_size=10)

    rate = rospy.Rate(10.0)  # 10hz

    sleep(2)

    ag.move_arm_to_initial_pose(pub_arm)
    state_dic["arm_initial_pose"] = 0

    while state_dic["arm_initial_pose"] == 0:
        time.sleep(0.1)

    print("Now Im at initial pose!")

    while state_dic["gripper_active"] == 0:
        time.sleep(0.1)

    print("Now the gripper is activated!")

    ag.gripper_close_fast(pub_gripper)

    while state_dic["gripper_closed"] == 0:
        time.sleep(0.1)

    print("gripper is closed!")

    # -----------------ARM MOVE ABOVE THE FINAL BOX------------------------------
    # arm_joints_goal = 0
    ag.move_arm_to_joints_state(pub_arm, 0.33516550064086914, -1.163656548862793, 1.3741701284991663,
                                           -1.8040539226927699, -1.5384181181537073, 0.3482170104980469)
    # ----------------------------------------------------------------------

    while state_dic["arm_joints_goal"] == 0:
        time.sleep(0.1)

    print("Now I arrived!")

    rospy.spin()
