#!/usr/bin/env python3
import rospy
import tf2_ros
from tf.transformations import euler_from_quaternion, euler_matrix


rospy.init_node("frame_change", anonymous=True)

tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

rate = rospy.Rate(1)


while not rospy.is_shutdown():
    try:
        trans = tfBuffer.lookup_transform('base_link', 'tool0_controller',  rospy.Time())
        x = trans.transform.rotation.x
        y = trans.transform.rotation.y
        z = trans.transform.rotation.z
        w = trans.transform.rotation.w

        euler = euler_from_quaternion([x, y, z, w], axes="rxyz")

        Re = euler_matrix(euler[0], euler[1], euler[2], 'rxyz')

        print(Re)

    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("error")
        pass

    rate.sleep()