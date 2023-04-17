#!/usr/bin/env python3

import json
import rospy
import time

from arm.srv import MoveArmToPoseGoal, MoveArmToPoseGoalRequest
from arm.srv import MoveArmToJointsState, MoveArmToJointsStateRequest
from arm.srv import StopArm
from std_msgs.msg import String


# from gripper.src.GripperStatusDTO import GripperStatusDTO


class ArmGripperComm:

    def __init__(self):
        self.state_dic = {"arm_moving": 0,
                          "activation_completed": False,
                          "gripper_closed": False,
                          "gripper_active": False,
                          "object_detected": False,
                          "gripper_pos": 0}

        rospy.Subscriber('gripper_response', String, self.gripper_response_callback)
        self.pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

        # set up arm controller service proxies
        rospy.wait_for_service('move_arm_to_pose_goal')
        rospy.wait_for_service('move_arm_to_joints_state')
        rospy.wait_for_service('stop_arm')
        self.move_arm_to_pose_goal_proxy = rospy.ServiceProxy('move_arm_to_pose_goal', MoveArmToPoseGoal)
        self.move_arm_to_joints_state_proxy = rospy.ServiceProxy('move_arm_to_joints_state', MoveArmToJointsState)
        self.stop_arm_proxy = rospy.ServiceProxy('stop_arm', StopArm)

    def gripper_response_callback(self, data):

        gripper_status_dict = json.loads(data.data)
        rospy.loginfo(gripper_status_dict)

        if gripper_status_dict["requested_position"] >= 140:
            self.state_dic["gripper_closed"] = False
        elif gripper_status_dict["requested_position"] <= 0:
            self.state_dic["gripper_closed"] = True

        self.state_dic["gripper_active"] = gripper_status_dict["is_activated"]
        self.state_dic["activation_completed"] = gripper_status_dict["activation_completed"]
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

    def gripper_init(self):
        """
        Sends a message to the gripper controller to initiate the gripper
        """
        my_dict_ = {'action': 'init'}
        encoded_data_string_ = json.dumps(my_dict_)
        rospy.loginfo(encoded_data_string_)
        self.pub_gripper.publish(encoded_data_string_)

        while not self.state_dic["activation_completed"]:
            time.sleep(0.1)

    def gripper_open_fast(self, wait=True):
        """
        Sends a message to the gripper controller to open the gripper
        """
        # values = [position, speed, force]
        my_dict = {'action': 'move', 'values': [0, 255, 0]}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

        if wait:
            while self.state_dic["gripper_closed"]:
                time.sleep(0.1)

    def gripper_close_fast(self, wait=True):
        """
        Sends a message to the gripper controller to close the gripper
        """
        # values = [position, speed, force]
        my_dict = {'action': 'move', 'values': [255, 255, 0]}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

        if wait:
            while not self.state_dic["gripper_closed"]:
                time.sleep(0.1)

    def gripper_status(self):
        # values = [position, speed, force]
        my_dict = {'action': 'status'}
        encoded_data_string = json.dumps(my_dict)
        rospy.loginfo(encoded_data_string)
        self.pub_gripper.publish(encoded_data_string)

    def move_arm_to_initial_pose(self, vel=0.1, a=0.1):
        """
        Sends a message to the arm controller to move the arm to a preconfigured initial position
        """
        vel = 1 if vel > 1 else 0.1 if vel < 0.1 else vel
        a = 1 if a > 1 else 0.1 if a < 0.1 else a

        return self.move_arm_to_joints_state(0.01725006103515625, -1.9415461025633753, 1.8129728476153772,
                                             -1.5927173099913539, -1.5878670851336878, 0.03150486946105957, vel, a)

    def move_arm_to_pose_goal(self, x, y, z, q1, q2, q3, q4, vel=0.1, a=0.1):
        """
        Sends a message to the arm controller to move the arm to a position based on the global frame
        """
        vel = 1 if vel > 1 else 0.1 if vel < 0.1 else vel
        a = 1 if a > 1 else 0.1 if a < 0.1 else a

        while self.state_dic["arm_moving"] != 0:
            time.sleep(0.1)

        self.state_dic["arm_moving"] == 1

        req = MoveArmToPoseGoalRequest(translation=(x, y, z), quaternions=(q1, q2, q3, q4),
                                       velocity=vel, acceleration=a)

        rospy.loginfo(req)

        try:
            resp = self.move_arm_to_pose_goal_proxy(req)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            return

        if self.state_dic["arm_moving"] == 1:
            self.state_dic["arm_moving"] == 0

        return resp

    def move_arm_to_joints_state(self, j1, j2, j3, j4, j5, j6, vel=0.1, a=0.1):
        """
        Sends a message to the arm controller to move the arm to a postion based on the arm's joints
        """
        vel = 1 if vel > 1 else 0.1 if vel < 0.1 else vel
        a = 1 if a > 1 else 0.1 if a < 0.1 else a

        while self.state_dic["arm_moving"] != 0:
            time.sleep(0.1)

        self.state_dic["arm_moving"] == 1

        req = MoveArmToJointsStateRequest(goal=[j1, j2, j3, j4, j5, j6], velocity=vel, acceleration=a)

        rospy.loginfo(req)

        try:
            resp = self.move_arm_to_joints_state_proxy(req)
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            return

        if self.state_dic["arm_moving"] == 1:
            self.state_dic["arm_moving"] == 0

        return resp

    def stop_arm(self):
        """
        Sends a message to the arm controller to stop the arm movement
        """
        self.state_dic["arm_moving"] == -1

        rospy.loginfo("stopping arm...")

        try:
            resp = self.stop_arm_proxy()
        except rospy.ServiceException as exc:
            print("Service did not process request: " + str(exc))
            return

        self.state_dic["arm_moving"] == 0

        return resp

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
