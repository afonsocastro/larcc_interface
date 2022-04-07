#!/usr/bin/env python3

import json
import time


# def wait_for_action_to_end(state, state1 = 1, state2 = 1, state3 = 1, state4 = 1):
def wait_for_action_to_end(state):
    while state == 0:
        time.sleep(0.1)


def gripper_open_fast(pub_gripper):
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [0, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    # rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)


def gripper_close_fast(pub_gripper):
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [255, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    # rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)


def move_arm_to_initial_pose(pub_arm):
    # global arm_initial_pose
    # arm_initial_pose = 0

    # -----------------ARM INITIAL POSE------------------------------
    arm_initial_pose_dict = {'action': 'move_to_initial_pose'}
    encoded_data_string_initial_pose = json.dumps(arm_initial_pose_dict)
    pub_arm.publish(encoded_data_string_initial_pose)
    # -------------------------------------------------------------------


def move_arm_to_pose_goal(pub_arm, x, y, z, q1, q2, q3, q4):
    # global arm_pose_goal
    # arm_pose_goal = 0

    _arm_dict = {'action': 'move_to_pose_goal', 'trans': [x, y, z],
                'quat': [q1, q2, q3, q4]}
    _encoded_data_string = json.dumps(_arm_dict)
    pub_arm.publish(_encoded_data_string)


def move_arm_to_joints_state(pub_arm, j1, j2, j3, j4, j5, j6):
    # global arm_joints_goal
    # arm_joints_goal = 0

    _arm_dict_ = {'action': 'move_to_joints_state',
                'joints': [j1, j2, j3, j4, j5, j6]}
    _encoded_data_string_ = json.dumps(_arm_dict_)
    pub_arm.publish(_encoded_data_string_)