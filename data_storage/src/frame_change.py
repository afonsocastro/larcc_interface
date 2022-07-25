#!/usr/bin/env python3
import rospy
import tf2_ros


rospy.init_node("frame_change", anonymous=True)

tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

rate = rospy.Rate(1)

while True:
    try:
        trans = tfBuffer.lookup_transform('base_link', 'shoulder_link',  rospy.Time())
        print(trans)
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("error")
        pass

    rate.sleep()