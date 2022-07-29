#!/usr/bin/env python3

import json
import time
import rospy
from std_msgs.msg import String
# from gripper.src.GripperStatusDTO import GripperStatusDTO


class ArmGripperComm:

    def __init__(self):
        self.state_dic = {"arm_initial_pose": 0,
                          "arm_pose_goal": 0,
                          "arm_joints_goal": 0,
                          "gripper_closed": False,
                          "gripper_active": False,
                          "object_detected": False,
                          "gripper_pos": 0}

        rospy.Subscriber("gripper_response", String, self.gripper_response_callback)
        self.pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

        rospy.Subscriber("arm_response", String, self.arm_response_callback)
        self.pub_arm = rospy.Publisher('arm_request', String, queue_size=10)

    def gripper_response_callback(self, data):

        gripper_status_dict = json.loads(data.data)
        rospy.loginfo(gripper_status_dict)

        if gripper_status_dict["requested_position"] >= 140:
            self.state_dic["gripper_closed"] = False
        elif gripper_status_dict["requested_position"] <= 0:
            self.state_dic["gripper_closed"] = True

        self.state_dic["gripper_active"] = gripper_status_dict["is_activated"]
        self.state_dic["object_detected"] = gripper_status_dict["object_detected"]
        self.state_dic["gripper_pos"] = gripper_status_dict["actual_position"]

        # if str(data).find("No object detected.") >= 0:
        #     self.state_dic["have_object"] = 0
        #     self.state_dic["gripper_closed"] = not self.state_dic["gripper_closed"]
        # elif str(data).find("Object detected.") >= 0:
        #     self.state_dic["have_object"] = 1
        #     self.state_dic["gripper_closed"] = not self.state_dic["gripper_closed"]
        # elif str(data).find("Gripper is now active! Ready to receive commands.") >= 0:
        #     self.state_dic["gripper_active"] = 1
        # elif str(data).find("status:") >= 0:
        #     self.state_dic["last_status"] = data

    def arm_response_callback(self, data):
        if str(data).find("Arm is now at initial pose.") >= 0:
            self.state_dic["arm_initial_pose"] = 1

            # if self.state_dic["gripper_active"] == 0:
            #     # -----------------GRIPPER ACTIVATION------------------------------
            #     my_dict_ = {'action': 'init'}
            #     encoded_data_string_ = json.dumps(my_dict_)
            #     rospy.loginfo(encoded_data_string_)
            #     self.pub_gripper.publish(encoded_data_string_)
            #     # ----------------------------------------------------------------------

        elif str(data).find("Arm is now at requested pose goal.") >= 0:
            self.state_dic["arm_pose_goal"] = 1
        elif str(data).find("Arm is now at requested joints state goal.") >= 0:
            self.state_dic["arm_joints_goal"] = 1

        # Sends a message to the gripper controller to open the gripper
    def gripper_open_fast(self):
        # values = [position, speed, force]
        my_dict = {'action': 'move', 'values': [0, 255, 0]}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

        while self.state_dic["gripper_closed"]:
            time.sleep(0.1)

    def gripper_connect(self):
        # values = [position, speed, force]
        my_dict = {'action': 'connect'}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

        while not self.state_dic["gripper_active"]:
            time.sleep(0.1)

    def gripper_disconnect(self):
        # values = [position, speed, force]
        my_dict = {'action': 'disconnect'}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

    # Sends a message to the gripper controller to initiate the gripper
    def gripper_init(self):
        my_dict_ = {'action': 'init'}
        encoded_data_string_ = json.dumps(my_dict_)
        rospy.loginfo(encoded_data_string_)
        self.pub_gripper.publish(encoded_data_string_)

        while not self.state_dic["gripper_active"]:
            time.sleep(0.1)

    # Sends a message to the gripper controller to close the gripper
    def gripper_close_fast(self):
        # values = [position, speed, force]
        my_dict = {'action': 'move', 'values': [255, 255, 0]}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

        while not self.state_dic["gripper_closed"]:
            time.sleep(0.1)

    def gripper_status(self):
        # values = [position, speed, force]
        my_dict = {'action': 'status'}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

    # Sends a message to the arm controller to move the arm to a preconfigured intial postion
    def move_arm_to_initial_pose(self):

        # -----------------ARM INITIAL POSE------------------------------
        arm_initial_pose_dict = {'action': 'move_to_initial_pose'}
        encoded_data_string_initial_pose = json.dumps(arm_initial_pose_dict)
        rospy.loginfo(arm_initial_pose_dict)
        self.pub_arm.publish(encoded_data_string_initial_pose)

        self.state_dic["arm_initial_pose"] = 0

        while self.state_dic["arm_initial_pose"] != 1:
            time.sleep(0.1)
        # -------------------------------------------------------------------

    # Sends a message to the arm controller to move the arm to a postion based on the global frame
    def move_arm_to_pose_goal(self, x, y, z, q1, q2, q3, q4):
        _arm_dict = {'action': 'move_to_pose_goal', 'trans': [x, y, z],
                     'quat': [q1, q2, q3, q4]}
        _encoded_data_string = json.dumps(_arm_dict)
        rospy.loginfo(_arm_dict)
        self.pub_arm.publish(_encoded_data_string)

        self.state_dic["arm_pose_goal"] = 0

        while self.state_dic["arm_pose_goal"] != 1:
            time.sleep(0.1)

    # Sends a message to the arm controller to move the arm to a postion based on the arm's joints
    def move_arm_to_joints_state(self, j1, j2, j3, j4, j5, j6):
        _arm_dict_ = {'action': 'move_to_joints_state',
                      'joints': [j1, j2, j3, j4, j5, j6]}
        _encoded_data_string_ = json.dumps(_arm_dict_)
        rospy.loginfo(_arm_dict_)
        self.pub_arm.publish(_encoded_data_string_)

        self.state_dic["arm_joints_goal"] = 0

        while self.state_dic["arm_joints_goal"] != 1:
            time.sleep(0.1)


# # Sends a message to the gripper controller to open the gripper
# def gripper_open_fast(pub_gripper):
#     # values = [position, speed, force]
#     my_dict = {'action': 'move', 'values': [0, 255, 0]}
#     encoded_data_string = json.dumps(my_dict)
#     rospy.loginfo(encoded_data_string)
#     pub_gripper.publish(encoded_data_string)

#
# # Sends a message to the gripper controller to initiate the gripper
# def gripper_init(pub_gripper):
#     my_dict_ = {'action': 'init'}
#     encoded_data_string_ = json.dumps(my_dict_)
#     rospy.loginfo(encoded_data_string_)
#     pub_gripper.publish(encoded_data_string_)


# # Sends a message to the gripper controller to close the gripper
# def gripper_close_fast(pub_gripper):
#     # values = [position, speed, force]
#     my_dict = {'action': 'move', 'values': [255, 255, 0]}
#     encoded_data_string = json.dumps(my_dict)
#     rospy.loginfo(encoded_data_string)
#     pub_gripper.publish(encoded_data_string)

#
# # Sends a message to the arm controller to move the arm to a preconfigured intial postion
# def move_arm_to_initial_pose(pub_arm):
#     # global arm_initial_pose
#     # arm_initial_pose = 0
#
#     print("Sending message")
#
#     # -----------------ARM INITIAL POSE------------------------------
#     arm_initial_pose_dict = {'action': 'move_to_initial_pose'}
#     encoded_data_string_initial_pose = json.dumps(arm_initial_pose_dict)
#     pub_arm.publish(encoded_data_string_initial_pose)
#     # -------------------------------------------------------------------


# Sends a message to the arm controller to move the arm to a postion based on the global frame
# def move_arm_to_pose_goal(pub_arm, x, y, z, q1, q2, q3, q4):
#     # global arm_pose_goal
#     # arm_pose_goal = 0
#
#     _arm_dict = {'action': 'move_to_pose_goal', 'trans': [x, y, z],
#                 'quat': [q1, q2, q3, q4]}
#     _encoded_data_string = json.dumps(_arm_dict)
#     pub_arm.publish(_encoded_data_string)


# # Sends a message to the arm controller to move the arm to a postion based on the arm's joints
# def move_arm_to_joints_state(pub_arm, j1, j2, j3, j4, j5, j6):
#     # global arm_joints_goal
#     # arm_joints_goal = 0
#
#     _arm_dict_ = {'action': 'move_to_joints_state',
#                 'joints': [j1, j2, j3, j4, j5, j6]}
#     _encoded_data_string_ = json.dumps(_arm_dict_)
#     pub_arm.publish(_encoded_data_string_)


# # Decodes the message sent from the arm controller to update the arm's state
# def arm_response(data, pub_gripper, state_dic):
#     if data == "Arm is now at initial pose.":
#         state_dic["arm_initial_pose"] = 1
#
#         if state_dic["gripper_active"] == 0:
#             # -----------------GRIPPER ACTIVATION------------------------------
#             my_dict_ = {'action': 'init'}
#             encoded_data_string_ = json.dumps(my_dict_)
#             # rospy.loginfo(encoded_data_string_)
#             pub_gripper.publish(encoded_data_string_)
#             # ----------------------------------------------------------------------
#
#     elif data == "Arm is now at requested pose goal.":
#         state_dic["arm_pose_goal"] = 1
#     elif data == "Arm is now at requested joints state goal.":
#         state_dic["arm_joints_goal"] = 1
#
#     return state_dic
#

# # Decodes the message sent from the gripper controller to update the gripper's state
# def gripper_response(data, state_dic):
#     if data == "No object detected.":
#         state_dic["have_object"] = 0
#         state_dic["gripper_closed"] = not state_dic["gripper_closed"]
#     elif data == "Object detected.":
#         state_dic["have_object"] = 1
#         state_dic["gripper_closed"] = not state_dic["gripper_closed"]
#     elif data == "Gripper is now active! Ready to receive commands.":
#         state_dic["gripper_active"] = 1
#
#     return state_dic
#
#
