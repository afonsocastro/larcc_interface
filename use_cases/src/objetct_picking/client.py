#!/usr/bin/env python
import socket
import rospy
import re
import tf

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 8080        # The port used by the server


def talker():
    rospy.init_node('client')
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(10) # 10hz
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print("before while not rospy.is_shutdown():")
        while not rospy.is_shutdown():
            print("after while not rospy.is_shutdown():")
            s.sendall(b'Hello, world')
            data = s.recv(1024)
            print("data: ", data)
            # print('Received', repr(data))
            # print(data.decode())
            x = re.findall(r"[-+]?\d*\.\d+|\d+", data.decode())
            trans = (float(x[0]), float(x[1]), float(x[2]))
            quat = (float(x[3]), float(x[4]), float(x[5]), float(x[6]))
            print('trans: ', trans)
            print('quat: ', quat)
            br.sendTransform(trans,
                             quat,
                             rospy.Time.now(),
                             "wood_block",
                             "camera_color_optical_frame")
            print("before rate.sleep()")
            rate.sleep()
            print("after rate.sleep()")


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass