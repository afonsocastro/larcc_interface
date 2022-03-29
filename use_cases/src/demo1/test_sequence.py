#!/usr/bin/env python3

from time import sleep
import rospy
from std_msgs.msg import String
import json
import tf
from transformation_t import *
import time


# ARM global states-------------------
arm_initial_pose = 0
arm_pose_goal = 0
arm_joints_goal = 0


# GRIPPER global states------------------------
gripper_open = 0
have_object = 0
gripper_active = 0


def gripper_response_callback(data):
    global arm_initial_pose
    global have_object
    global gripper_open
    global gripper_active

    if data.data == "No object detected.":
        have_object = 0
        gripper_open = not gripper_open
    elif data.data == "Object detected.":
        have_object = 1
        gripper_open = not gripper_open
    elif data.data == "Gripper is now active! Ready to receive commands.":
        gripper_active = 1


def arm_response_callback(data):
    global arm_initial_pose
    global arm_pose_goal
    global arm_joints_goal
    global gripper_active

    if data.data == "Arm is now at initial pose.":
        arm_initial_pose = 1

        if gripper_active == 0:
            # -----------------GRIPPER ACTIVATION------------------------------
            my_dict_ = {'action': 'init'}
            encoded_data_string_ = json.dumps(my_dict_)
            rospy.loginfo(encoded_data_string_)
            pub_gripper.publish(encoded_data_string_)
            # ----------------------------------------------------------------------

    elif data.data == "Arm is now at requested pose goal.":
        arm_pose_goal = 1
    elif data.data == "Arm is now at requested joints state goal.":
        arm_joints_goal = 1


def wait_for_action_to_end(state):
    while state == 0:
        time.sleep(0.1)


def gripper_open_fast():
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [0, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    # rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)


def gripper_close_fast():
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [255, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    # rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)


def move_arm_to_initial_pose():
    global arm_initial_pose
    arm_initial_pose = 0

    # -----------------ARM INITIAL POSE------------------------------
    arm_initial_pose_dict = {'action': 'move_to_initial_pose'}
    encoded_data_string_initial_pose = json.dumps(arm_initial_pose_dict)
    pub_arm.publish(encoded_data_string_initial_pose)
    # -------------------------------------------------------------------


def move_arm_to_pose_goal(x, y, z, q1, q2, q3, q4):
    global arm_pose_goal
    arm_pose_goal = 0

    _arm_dict = {'action': 'move_to_pose_goal', 'trans': [x, y, z],
                'quat': [q1, q2, q3, q4]}
    _encoded_data_string = json.dumps(_arm_dict)
    pub_arm.publish(_encoded_data_string)


def move_arm_to_joints_state(j1, j2, j3, j4, j5, j6):
    global arm_joints_goal
    arm_joints_goal = 0

    _arm_dict_ = {'action': 'move_to_joints_state',
                'joints': [j1, j2, j3, j4, j5, j6]}
    _encoded_data_string_ = json.dumps(_arm_dict_)
    pub_arm.publish(_encoded_data_string_)


if __name__ == '__main__':
    rospy.init_node('test_sequence', anonymous=True)

    rospy.Subscriber("gripper_response", String, gripper_response_callback)
    pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

    rospy.Subscriber("arm_response", String, arm_response_callback)
    pub_arm = rospy.Publisher('arm_request', String, queue_size=10)

    rate = rospy.Rate(10.0)  # 10hz

    sleep(2)

    move_arm_to_initial_pose()

    while arm_initial_pose == 0:
        time.sleep(0.1)

    print("Now Im at initial pose!")

    while gripper_active == 0:
        time.sleep(0.1)

    print("Now the gripper is activated!")

    # -----------------ARM MOVE ABOVE THE FINAL BOX------------------------------
    # arm_joints_goal = 0
    move_arm_to_joints_state(0.33516550064086914, -1.163656548862793, 1.3741701284991663,
                                           -1.8040539226927699, -1.5384181181537073, 0.3482170104980469)
    # ----------------------------------------------------------------------

    while arm_joints_goal == 0:
        time.sleep(0.1)

    print("Now I arrived!")

    rospy.spin()
