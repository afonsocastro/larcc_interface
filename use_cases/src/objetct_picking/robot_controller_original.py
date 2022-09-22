#! /usr/bin/python3

import numpy as np
import rospy
import tf
from lib.src.transformation_t import TransformationT
import time


def main():

    listener = tf.TransformListener()

    br = tf.TransformBroadcaster()
    wood_block_T_tool0_controller = TransformationT("wood_block", "tool0_controller")
    z_offset = 0.3
    y_offset = 0.04
    x_offset = 0.005
    fixed_trans = [0.03 + x_offset , -0.02 + y_offset, 0.15 + z_offset]

    wood_block_T_tool0_controller.setTranslation(fixed_trans)

    fixed_rot = [[-1.0000000, 0.0000000, 0.0000000],
           [0.0000000, 1.0000000, 0.0000000],
           [0.0000000, 0.0000000, -1.0000000]]

    wood_block_T_tool0_controller.setRotation(fixed_rot)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        time.sleep(0.5)
        try:
            (trans, rot) = listener.lookupTransform('/wood_block', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("WAS NOT ABLE TO GET wood_block to base_link TF")
            continue

        base_link_T_wood_block = TransformationT("base_link", "wood_block")
        base_link_T_wood_block.setTranslation(trans)
        base_link_T_wood_block.setQuaternion(rot)

        base_link_T_tool0_controller = TransformationT("base_link", "tool0_controller")

        base_link_T_tool0_controller.matrix = np.dot(wood_block_T_tool0_controller.matrix, base_link_T_wood_block.matrix)

        base_link_T_goal = TransformationT("goal2", "base_link")
        base_link_T_goal.matrix = np.linalg.inv(base_link_T_tool0_controller.matrix)
        goal_trans = base_link_T_goal.getTranslation()
        goal_quat = base_link_T_goal.getQuaternion()

        tutorial.go_to_pose_goal(goal_trans[0], goal_trans[1], goal_trans[2], goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3])

        rate.sleep()


if __name__ == '__main__':
    main()
