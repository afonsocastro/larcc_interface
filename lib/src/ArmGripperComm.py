#!/usr/bin/env python3

import json
import time
import rospy


# Sends a message to the gripper controller to open the gripper
def gripper_open_fast(pub_gripper):
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [0, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)


# Sends a message to the gripper controller to initiate the gripper
def gripper_init(pub_gripper):
    my_dict_ = {'action': 'init'}
    encoded_data_string_ = json.dumps(my_dict_)
    rospy.loginfo(encoded_data_string_)
    pub_gripper.publish(encoded_data_string_)


# Sends a message to the gripper controller to close the gripper
def gripper_close_fast(pub_gripper):
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [255, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)


# Sends a message to the arm controller to move the arm to a preconfigured intial postion
def move_arm_to_initial_pose(pub_arm):
    # global arm_initial_pose
    # arm_initial_pose = 0

    print("Sending message")

    # -----------------ARM INITIAL POSE------------------------------
    arm_initial_pose_dict = {'action': 'move_to_initial_pose'}
    encoded_data_string_initial_pose = json.dumps(arm_initial_pose_dict)
    pub_arm.publish(encoded_data_string_initial_pose)
    # -------------------------------------------------------------------


# Sends a message to the arm controller to move the arm to a postion based on the global frame
def move_arm_to_pose_goal(pub_arm, x, y, z, q1, q2, q3, q4):
    # global arm_pose_goal
    # arm_pose_goal = 0

    _arm_dict = {'action': 'move_to_pose_goal', 'trans': [x, y, z],
                'quat': [q1, q2, q3, q4]}
    _encoded_data_string = json.dumps(_arm_dict)
    pub_arm.publish(_encoded_data_string)


# Sends a message to the arm controller to move the arm to a postion based on the arm's joints
def move_arm_to_joints_state(pub_arm, j1, j2, j3, j4, j5, j6):
    # global arm_joints_goal
    # arm_joints_goal = 0

    _arm_dict_ = {'action': 'move_to_joints_state',
                'joints': [j1, j2, j3, j4, j5, j6]}
    _encoded_data_string_ = json.dumps(_arm_dict_)
    pub_arm.publish(_encoded_data_string_)


# Decodes the message sent from the arm controller to update the arm's state
def arm_response(data, pub_gripper, state_dic):
    if data == "Arm is now at initial pose.":
        state_dic["arm_initial_pose"] = 1

        if state_dic["gripper_active"] == 0:
            # -----------------GRIPPER ACTIVATION------------------------------
            my_dict_ = {'action': 'init'}
            encoded_data_string_ = json.dumps(my_dict_)
            # rospy.loginfo(encoded_data_string_)
            pub_gripper.publish(encoded_data_string_)
            # ----------------------------------------------------------------------

    elif data == "Arm is now at requested pose goal.":
        state_dic["arm_pose_goal"] = 1
    elif data == "Arm is now at requested joints state goal.":
        state_dic["arm_joints_goal"] = 1

    return state_dic


# Decodes the message sent from the gripper controller to update the gripper's state
def gripper_response(data, state_dic):
    if data == "No object detected.":
        state_dic["have_object"] = 0
        state_dic["gripper_closed"] = not state_dic["gripper_closed"]
    elif data == "Object detected.":
        state_dic["have_object"] = 1
        state_dic["gripper_closed"] = not state_dic["gripper_closed"]
    elif data == "Gripper is now active! Ready to receive commands.":
        state_dic["gripper_active"] = 1

    return state_dic

