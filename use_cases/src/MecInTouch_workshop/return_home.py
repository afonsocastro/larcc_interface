#!/usr/bin/env python3

import rospy  # Import ROS Python library
from UR10eArm import UR10eArm
from RobotiqHand import RobotiqHand


if __name__ == '__main__':

    rospy.init_node('return_home', anonymous=True)  # Initialize the node
    rate = rospy.Rate(1)
    manipulator = UR10eArm()
    hand = RobotiqHand()
    hand.connect("192.168.56.2", 54321)
    hand.reset()
    hand.activate()

    # shoulder_pan, shoulder lift, elbow, wrist1, wrist2, wrist3
    manipulator.go_to_joint_state(0, -1.392, 1.372, -1.591, -1.511, 0, 1, 1)

    hand.move(0, 255, 1)

    hand.disconnect()
