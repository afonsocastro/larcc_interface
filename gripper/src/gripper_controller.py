#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from larcc_classes.gripper.RobotiqHand import RobotiqHand
import json

# -*- coding: utf-8 -*-
HOST = "192.168.56.2"
PORT = 54321

# --------------------------------------------------------------------
# -------------------------Initialization-----------------------------
# --------------------------------------------------------------------
hand = RobotiqHand()


def request_gripper_callback(data):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("I heard %s", data.data)

    gripper_request_dict = json.loads(data.data)

    if gripper_request_dict['action'] == 'init':
        init_state = hand_init_procedures()
        if init_state == 0:
            result = 'Gripper is not correctly activated. Disconnecting.'
            print(result)
        elif init_state == 1:
            result = 'Gripper is now active! Ready to receive commands.'
            print(result)
        # pub.publish(result)

    elif gripper_request_dict['action'] == 'move':
        position = gripper_request_dict['values'][0]
        speed = gripper_request_dict['values'][1]
        force = gripper_request_dict['values'][2]

        status, position_mm, force_mA = move(position, speed, force)

        # if status == 0:
        #     # result = 'no object detected: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA)
        #     result = 'No object detected.'
        #     print(result)
        # elif status == 1:
        #     # result = 'object detected closing: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA)
        #     result = 'Object detected.'
        #     print(result)
        # elif status == 2:
        #     # result = 'object detected opening: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA)
        #     result = 'Object detected.'
        #     print(result)
        # else:
        #     result = 'failed'
        #     print('status:' + str(result))

        # pub.publish(result)
    # elif gripper_request_dict["action"] == 'status':
        # result = hand.get_instant_gripper_status()
        # my_json_string = dto_to_json(result)
        # pub.publish(my_json_string)

    elif gripper_request_dict['action'] == 'connect':
        hand.connect(HOST, PORT)
        # result = 'Gripper is connected.'
        # pub.publish(result)
        # result = hand.get_instant_gripper_status()
        # my_json_string = dto_to_json(result)
        # pub.publish(my_json_string)

    result = hand.get_instant_gripper_status()
    my_json_string = dto_to_json(result)
    pub.publish(my_json_string)

    if gripper_request_dict['action'] == 'disconnect':
        # result = hand.get_instant_gripper_status()
        # my_json_string = dto_to_json(result)
        # pub.publish(my_json_string)
        hand.disconnect()
        # result = 'Gripper is disconnected.'
        # pub.publish(result)


def hand_init_procedures():
    # hand.connect(HOST, PORT)

    hand.reset()
    hand.activate()
    result = hand.wait_activate_complete()
    if result != 0x31:
        hand.disconnect()
        return 0
    hand.adjust()
    return 1


def move(position, speed, force):
    hand.move(position, speed, force)
    (status, final_position, final_force) = hand.wait_move_complete()

    position_mm = hand.get_position_mm(final_position)
    force_mA = hand.get_force_mA(final_force)

    return status, position_mm, force_mA


def dto_to_json(dto):
    my_dic = dto.__dict__
    my_json = json.dumps(my_dic)

    return my_json


if __name__ == '__main__':
    # hand = RobotiqHand()

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('gripper_controller', anonymous=True)
    pub = rospy.Publisher('gripper_response', String, queue_size=10)
    rospy.Subscriber("gripper_request", String, request_gripper_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
