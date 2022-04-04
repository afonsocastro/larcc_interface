#!/usr/bin/env python3
import json
import rospy


def gripper_open_fast(pub_gripper):
    # values = [position, speed, force]
    my_dict = {'action': 'move', 'values': [0, 255, 0]}
    encoded_data_string = json.dumps(my_dict)
    rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)