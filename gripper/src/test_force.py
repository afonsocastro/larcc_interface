#!/usr/bin/env python3

import time
import signal
from RobotiqHand import RobotiqHand

HOST = "192.168.56.2"
PORT = 54321

cont = True


def handler(signal, frame):
    global cont
    cont = False


def test_robotiq():

    print('test_force start')
    hand = RobotiqHand()
    hand.connect(HOST, PORT)

    try:
        print('activate: start')
        hand.reset()
        hand.activate()
        result = hand.wait_activate_complete()
        print('activate: result = 0x{:02x}'.format(result))
        if result != 0x31:
            hand.disconnect()
            return
        print('adjust: start')
        hand.adjust()
        print('adjust: finish')

        time.sleep(1)

        while cont:
            time.sleep(0.05)
            status = hand.get_instant_gripper_status()

    except:
        print('Ctrl-c pressed')
        # TODO create handler to close the door

    hand.disconnect()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    test_robotiq()
