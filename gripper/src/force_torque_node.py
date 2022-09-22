#!/usr/bin/env python3

import rospy
import time
import signal
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from gripper.msg import ForceTorque
from larcc_classes.gripper.RobotiqHand import RobotiqHand

HOST = "192.168.56.2"
PORT = 54321

cont = True


class ForceTorqueValues:
    def __init__(self):
        self.wrench_force = []
        self.wrench_torque = []
        self.joints_effort = []
        self.gripper_current = 0

    def joint_state_callback(self, data):
        self.joints_effort = data.effort

    def wrench_callback(self, data):
        self.wrench_force = data.wrench.force
        self.wrench_torque = data.wrench.torque


def handler(signal, frame):
    global cont
    cont = False


def gripper_init(hand_robot):
    print('activate: start')
    hand_robot.reset()
    hand_robot.activate()
    result = hand_robot.wait_activate_complete()
    print('activate: result = 0x{:02x}'.format(result))
    if result != 0x31:
        hand_robot.disconnect()
    print('adjust: start')
    hand_robot.adjust()
    print('adjust: finish')

    return hand_robot


if __name__ == '__main__':

    rospy.init_node('force_torque_node', anonymous=True)

    print("Node initiated")

    # rate = rospy.Rate(10.0)  # 10hz
    rate = rospy.Rate(5000.0)  # 10hz

    force_torque_values = ForceTorqueValues()

    pub_force_trorque = rospy.Publisher('force_torque_values', ForceTorque, queue_size=1000)

    rospy.Subscriber("joint_states", JointState, force_torque_values.joint_state_callback)
    rospy.Subscriber("wrench", WrenchStamped, force_torque_values.wrench_callback)

    msg = ForceTorque()

    signal.signal(signal.SIGINT, handler)
    hand = RobotiqHand()
    hand.connect(HOST, PORT)

    try:
        hand = gripper_init(hand)

        start = time.time()
        closed = 0

        while not rospy.is_shutdown():

            status = hand.get_instant_gripper_status()
            force_torque_values.gripper_current = status.actual_force_motor_current

            msg.WrenchForce = force_torque_values.wrench_force
            msg.WrenchTorque = force_torque_values.wrench_torque
            msg.JointsEffort.data = force_torque_values.joints_effort
            msg.GripperCurrent.data = force_torque_values.gripper_current

            pub_force_trorque.publish(msg)

            end = time.time()
            if end - start > 2 and closed == 0:
                print('close slow')
                hand.move(255, 0, 1)
                closed = 1

            # now = rospy.get_rostime()
            # rospy.loginfo("Current time %i %i", now.secs, now.nsecs)

            rate.sleep()
        rospy.spin()

    except:
        print('Ctrl-c pressed')

    hand.disconnect()
