#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import time
import signal
from larcc_classes.gripper.RobotiqHand import RobotiqHand
from larcc_classes.gripper.GripperStatusDTO import GripperStatusDTO

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
    print ('test_robotiq start')
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
        print('close slow')
        hand.move(255, 0, 1)
        (status, position, force) = hand.wait_move_complete()

        time.sleep(4)
        print('open fast')
        hand.move(0, 255, 0)
        (status, position, force) = hand.wait_move_complete()

        gripper_status = GripperStatusDTO(name=True, location=42)
        print(gripper_status)

        # while True:
        #     time.sleep(0.1)
        #     print('close fast')
        #     hand.move(255, 255, 1)
        #     (status, position, force) = hand.wait_move_complete()
        #
        #     time.sleep(0.1)
        #     print('open fast')
        #     hand.move(0, 255, 0)
        #     (status, position, force) = hand.wait_move_complete()



    #     while cont:
    #         print ('close slow')
    #         hand.move(255, 0, 1)
    #         (status, position, force) = hand.wait_move_complete()
    #         position_mm = hand.get_position_mm(position)
    #         force_mA = hand.get_force_mA(force)
    #
    #         if status == 0:
    #             print( 'no object detected: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
    #         elif status == 1:
    #             print('object detected closing: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
    #             print('keeping')
    #             time.sleep(5)
    #         elif status == 2:
    #             print('object detected opening: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
    #         else:
    #             print('failed')
    #
    #         print('open fast')
    #         hand.move(0, 255, 0)
    #         (status, position, force) = hand.wait_move_complete()
    #
    #         print('test close')
    #         hand.close()
    #         (status, position, force) = hand.wait_move_complete()
    #
    #         print('re-open fast')
    #         hand.move(0, 255, 0)
    #
    #         print('close fast')
    #         hand.move(255, 255, 0)
    #         (status, position, force) = hand.wait_move_complete()
    #
    #         # if keyboard.is_pressed('q'):  # if key 'q' is pressed
    #         #     print('You Pressed A Key!')
    #
    #         (status, position, force) = hand.wait_move_complete()
    #         position_mm = hand.get_position_mm(position)
    #         force_mA = hand.get_force_mA(force)
    #         print('position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
    except:
        print('Ctrl-c pressed')
        #TODO create handler to close the door

    hand.disconnect()



if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    test_robotiq()
