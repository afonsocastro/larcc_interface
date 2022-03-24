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

    while not rospy.is_shutdown():
        sleep(0.5)

        listener.waitForTransform('/wood_block', '/camera_color_optical_frame', rospy.Time(), rospy.Duration(4.0))
        print("just passe the waitForTransform. Going now to lookupTransform")
        try:
            (trans, rot) = listener.lookupTransform('/wood_block', '/camera_color_optical_frame', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("WAS NOT ABLE TO GET wood_block to base_link TF")
            continue


        rate.sleep()
    rospy.spin()
