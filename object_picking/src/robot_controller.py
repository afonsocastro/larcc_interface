#! /usr/bin/python3
import numpy as np

from move_group_python_interface_tutorial import *
import rospy
import time
import tf
from transformation_t import *


def move_robot(tutorial, centroid_coord, is_far):
    px = centroid_coord[0]
    pz = centroid_coord[1]
    # x = 0.04 + (px / 2000)
    x = round(-0.8 + (px / 2000), 3)
    z = round(1.7 - (pz / 2000), 3)
    print("\nNEXT GOAL:")
    print(x, z)

    # MOVING THE ROBOT
    tutorial.go_to_pose_goal(x, 1, z, is_far)


def move_robot_initial_position(tutorial):
    # move_robot(tutorial, (0.36, 1.36), True)
    tutorial.go_to_joint_state(0.01725006103515625, -1.9415461025633753, 1.8129728476153772, -1.5927173099913539, -1.5878670851336878, 0.03150486946105957)


def main():

    # diagonal_size = 1468.60478
    # step = 700
    # --------------------------------------------------------------------
    # -------------------------initialization-----------------------------
    # --------------------------------------------------------------------
    tutorial = MoveGroupPythonInterfaceTutorial()
    move_group = tutorial.move_group
    # move_robot_initial_position(tutorial)
    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()
    wood_block_T_tool0_controller = TransformationT("wood_block", "tool0_controller")
    # fixed_trans = [0.03, -0.02, 0.2]
    # z_offset = 0.243
    z_offset = 0.3

    y_offset = 0.04
    x_offset = 0.005
    fixed_trans = [0.03 + x_offset , -0.02 + y_offset, 0.15 + z_offset]
    # fixed_trans = [0.0000000, 0.0000000, 0.0000000]
    wood_block_T_tool0_controller.setTranslation(fixed_trans)

    fixed_rot = [[-1.0000000, 0.0000000, 0.0000000],
           [0.0000000, 1.0000000, 0.0000000],
           [0.0000000, 0.0000000, -1.0000000]]

    wood_block_T_tool0_controller.setRotation(fixed_rot)

    fixed_quat = wood_block_T_tool0_controller.getQuaternion()

    # print("\nwood_block_T_tool0_controller:\n")
    # print(wood_block_T_tool0_controller)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():

        # input("Press Enter to continue...")
        time.sleep(0.5)
        try:
            # now = rospy.get_rostime()
            # exists_transform = listener.waitForTransform('/wood_block', '/base_link', now, rospy.Duration(10))
            # print("\nexists_transform: ", exists_transform)
            (trans, rot) = listener.lookupTransform('/wood_block', '/base_link', rospy.Time(0))
            print("\nYYEEEEESSSSSS")
            print("\ntrans\n")
            print(trans)
            print("\nrot\n")
            print(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("WAS NOT ABLE TO GET wood_block to base_link TF")
            continue

        base_link_T_wood_block = TransformationT("base_link", "wood_block")
        base_link_T_wood_block.setTranslation(trans)
        base_link_T_wood_block.setQuaternion(rot)

        # print("\nbase_link_T_wood_block:\n")
        # print(base_link_T_wood_block)

        base_link_T_tool0_controller = TransformationT("base_link", "tool0_controller")

        base_link_T_tool0_controller.matrix = np.dot(wood_block_T_tool0_controller.matrix, base_link_T_wood_block.matrix)
        new_trans = base_link_T_tool0_controller.getTranslation()
        new_quat = base_link_T_tool0_controller.getQuaternion()

        base_link_T_goal = TransformationT("goal2", "base_link")
        base_link_T_goal.matrix = np.linalg.inv(base_link_T_tool0_controller.matrix)
        new2_trans = base_link_T_goal.getTranslation()
        new2_quat = base_link_T_goal.getQuaternion()

        # # trans = (float(x[0]), float(x[1]), float(x[2]))
        # # new_quat = (float(0), float(0), float(0), float(0))

        # print('new2_trans: ', new2_trans)
        # print('new2_quat: ', new2_quat)

        # # print('fixed_trans: ', fixed_trans)
        # # print('fixed_quat: ', fixed_quat)
        #

        # current_pose = move_group.get_current_pose().pose
        # print(current_pose)
        tutorial.go_to_pose_goal(new2_trans[0], new2_trans[1], new2_trans[2], new2_quat[0], new2_quat[1], new2_quat[2], new2_quat[3])
        rate.sleep()


if __name__ == '__main__':
    main()
