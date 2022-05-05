#!/usr/bin/env python3

from time import sleep
import rospy
from std_msgs.msg import String
import json
import tf
from transformation_t import *
import time

if __name__ == '__main__':
    rospy.init_node('test', anonymous=True)
    rate = rospy.Rate(10.0)  # 10hz
    listener = tf.TransformListener()
    no_tf_counter = 0
    while not rospy.is_shutdown():
        sleep(1)

        # listener.waitForTransform('/wood_block', '/camera_color_optical_frame', rospy.Time(), rospy.Duration(10.0))
        # print("just passed the waitForTransform. Going now to lookupTransform")

        try:
            (trans, rot) = listener.lookupTransform('/wood_block', '/camera_color_optical_frame', rospy.Time(0))
            no_tf_counter = 0
            print("Yep, I have the TF.")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            no_tf_counter += 1
            print("WAS NOT ABLE TO GET wood_block to base_link TF")
            if no_tf_counter == 8:
                print("MOVING UP!!!")
            continue

        listener.clear()
        rate.sleep()

