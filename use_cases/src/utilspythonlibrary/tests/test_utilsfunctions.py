#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
# from utilspythonlib import utilsfunctions as uf
import json
import sys

import use_cases.src.utils.ArmGripperLib as ag



# import os
#
# print(os.environ.get('PYTHONPATH'))

# sys.path.append('/home/joel/catkin_ws/src/larcc_interface/use_cases/')

if __name__ == "__main__":
    rospy.init_node('test1', anonymous=True)

    pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

    rate = rospy.Rate(1.0)
    # values = [position, speed, force]

    for path in sys.path:
        print(path)

    # uf.gripper_open_fast(pub_gripper)

    while not rospy.is_shutdown():
        ag.gripper_open_fast(pub_gripper)
        rate.sleep()