#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import signal
from RobotiqHand import RobotiqHand

#------------------------------------------------------------------------------
# test_robotiq.py
#------------------------------------------------------------------------------
# HOST = "192.168.102.216"
HOST = "192.168.56.2"
PORT = 54321

cont = True


def handler(signal, frame):
    global cont
    cont = False


def test_robotiq():
    print ('test_force start')
    hand = RobotiqHand()
    hand.connect(HOST, PORT)

    try:
        print ('activate: start')
        hand.reset()
        hand.activate()
        result = hand.wait_activate_complete()
        print ('activate: result = 0x{:02x}'.format(result))
        if result != 0x31:
            hand.disconnect()
            return
        print ('adjust: start')
        hand.adjust()
        print ('adjust: finish')

        time.sleep(4)
        print('open fast')
        hand.move(0, 255, 0)
        (status, position, force) = hand.wait_move_complete()

        time.sleep(4)
        print('close slow')
        hand.move(255, 0, 1)
        # (status, position, force) = hand.wait_move_complete()

        while cont:
            time.sleep(0.1)
            # (status, position, force) = hand.wait_move_complete()
            (position, force) = hand.get_instant_status()
            position_mm = hand.get_position_mm(position)
            force_mA = hand.get_force_mA(force)
            print('instant: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))

            #
            # if status == 0:
            #     print( 'no object detected: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
            # elif status == 1:
            #     print('object detected closing: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
            #     print('keeping')
            #     # time.sleep(5)
            # elif status == 2:
            #     print('object detected opening: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
            # else:
            #     print('failed')

    except:
        print('Ctrl-c pressed')
        #TODO create handler to close the door

    hand.disconnect()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    test_robotiq()
