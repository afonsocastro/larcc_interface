#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
# from utilspythonlib import utilsfunctions as uf
import json
import sys

from lib import ArmGripperComm as ag

print("All imported")

if __name__ == "__main__":
    rospy.init_node('test1', anonymous=True)

    pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

    rate = rospy.Rate(1.0)
    # values = [position, speed, force]
    # uf.gripper_open_fast(pub_gripper)

    while not rospy.is_shutdown():
        print("In the loop!!!")
        ag.gripper_open_fast(pub_gripper)
        rate.sleep()