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
    elif data.data == "Object detected.":
        have_object = 1
    elif data.data == "Gripper is now active! Ready to receive commands.":
        gripper_active = 1

    gripper_open = not gripper_open


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


def wait_for_arm_move(state):
    while state == 0:
        time.sleep(0.1)


def gripper_open_fast():
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [0, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)


def gripper_close_fast():
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [255, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    rospy.loginfo(encoded_data_string)
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


def pick_and_place(trans_goal, quat_goal):
    global arm_pose_goal
    # -----------------ARM MOVE DOWN TO WOOD-BLOCK------------------------------
    move_arm_to_pose_goal(trans_goal[0], trans_goal[1], trans_goal[2] - 0.1, quat_goal[0], quat_goal[1], quat_goal[2],
                          quat_goal[3])
    # ----------------------------------------------------------------------

    sleep(1)

    # while arm_pose_goal == 0:
    #     sleep(0.1)

    gripper_close_fast()
    sleep(1)

    # -----------------ARM MOVE UP------------------------------
    move_arm_to_pose_goal(trans_goal[0], trans_goal[1], trans_goal[2] + 0.2, quat_goal[0], quat_goal[1], quat_goal[2],
                          quat_goal[3])
    # ----------------------------------------------------------------------

    sleep(1)

    # -----------------ARM MOVE ABOVE THE FINAL BOX------------------------------
    move_arm_to_joints_state(0.33516550064086914, -1.163656548862793, 1.3741701284991663, -1.8040539226927699,
                             -1.5384181181537073, 0.3482170104980469)
    # ----------------------------------------------------------------------

    sleep(2)

    # -----------------ARM MOVE TO FINAL BOX (JUST DOWN)------------------------------
    move_arm_to_joints_state(0.34529924392700195, -1.0220564168742676, 1.5045183340655726, -2.004709859887594,
                             -1.5726912657367151, 0.3451671600341797)
    # ----------------------------------------------------------------------

    sleep(3)
    gripper_open_fast()
    sleep(1)

    # -----------------ARM MOVE ABOVE THE FINAL BOX------------------------------
    move_arm_to_joints_state(0.33516550064086914, -1.163656548862793, 1.3741701284991663, -1.8040539226927699,
                             -1.5384181181537073, 0.3482170104980469)
    # ----------------------------------------------------------------------


if __name__ == '__main__':
    rospy.init_node('demo1', anonymous=True)

    rospy.Subscriber("gripper_response", String, gripper_response_callback)
    pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

    rospy.Subscriber("arm_response", String, arm_response_callback)
    pub_arm = rospy.Publisher('arm_request', String, queue_size=10)

    rate = rospy.Rate(10.0)  # 10hz

    sleep(10)
    move_arm_to_initial_pose()

    listener = tf.TransformListener()

    br = tf.TransformBroadcaster()
    wood_block_T_tool0_controller = TransformationT("wood_block", "tool0_controller")
    z_offset = 0.3

    # y_offset = 0.04
    y_offset = 0.0

    x_offset = -0.03
    # x_offset = 0.005
    fixed_trans = [0.03 + x_offset, -0.02 + y_offset, 0.15 + z_offset]

    wood_block_T_tool0_controller.setTranslation(fixed_trans)

    fixed_rot = [[-1.0000000, 0.0000000, 0.0000000],
                 [0.0000000, 1.0000000, 0.0000000],
                 [0.0000000, 0.0000000, -1.0000000]]

    wood_block_T_tool0_controller.setRotation(fixed_rot)

    first_warning = 0

    while arm_initial_pose == 0:
        time.sleep(0.1)

    start_time = time.time()
    while not rospy.is_shutdown():
        sleep(0.5)

        try:
            listener.waitForTransform('/wood_block', '/base_link', rospy.Time(0), rospy.Duration(30.0))
            (trans, rot) = listener.lookupTransform('/wood_block', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("WAS NOT ABLE TO GET wood_block to base_link TF")
            continue

        base_link_T_wood_block = TransformationT("base_link", "wood_block")
        base_link_T_wood_block.setTranslation(trans)
        base_link_T_wood_block.setQuaternion(rot)

        base_link_T_tool0_controller = TransformationT("base_link", "tool0_controller")

        base_link_T_tool0_controller.matrix = np.dot(wood_block_T_tool0_controller.matrix,
                                                     base_link_T_wood_block.matrix)

        base_link_T_goal = TransformationT("goal2", "base_link")
        base_link_T_goal.matrix = np.linalg.inv(base_link_T_tool0_controller.matrix)
        goal_trans = base_link_T_goal.getTranslation()
        goal_quat = base_link_T_goal.getQuaternion()

        # -----------------ARM MOVE TO WOOD-BLOCK------------------------------
        # sleep(5)
        arm_dict = {'action': 'move_to_pose_goal', 'trans': [goal_trans[0], goal_trans[1], goal_trans[2]],
                    'quat': [goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3]]}
        encoded_data_string = json.dumps(arm_dict)
        pub_arm.publish(encoded_data_string)
        # ----------------------------------------------------------------------
        #
        now_time = time.time()

        time_elapsed = (now_time - start_time)

        if time_elapsed > 20 and first_warning == 0:
            first_warning = 1
            gripper_close_fast()
            gripper_open_fast()

        if time_elapsed > 25 and first_warning == 1:
            first_warning = 0
            pick_and_place(goal_trans, goal_quat)
            sleep(1)
            move_arm_to_initial_pose()
            while arm_initial_pose == 0:
                time.sleep(0.1)

            # Instantly delete all tf tree (Clear all data)
            listener.clear()
            start_time = time.time()

        rate.sleep()
    rospy.spin()
