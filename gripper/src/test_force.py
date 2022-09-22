#!/usr/bin/env python3

import signal
from larcc_classes.gripper.RobotiqHand import RobotiqHand

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
        # time.sleep(5)

        while cont:
            print('close slow')
            hand.move(255, 0, 1)
            (status, position, force) = hand.wait_move_complete()

    except:
        print('Ctrl-c pressed')
        # TODO create handler to close the door

    hand.disconnect()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    test_robotiq()
