#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import signal
from RobotiqHand import RobotiqHand
from ast import literal_eval
import binascii

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


def splitting(char_array):
    spaced_chars = ""
    for i, c in enumerate(char_array):
      if i % 2 == 1:
        spaced_chars += c + " "
      else:
        spaced_chars += c

    return spaced_chars


def hex2dec(hex_byte):
    hex_string = str(hex(hex_byte))
    dec = literal_eval(hex_string)
    return dec


def hexbytearray2decbytearray(hexbytearray):
    decbytearray = []
    for i in hexbytearray:
        new_i = hex2dec(i)
        decbytearray.append(new_i)
    return decbytearray


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

        # time.sleep(2)
        # print('open fast')
        # hand.move(0, 255, 0)
        # (status, position, force) = hand.wait_move_complete()

        time.sleep(2)
        print('close slow')
        hand.move(0xff, 0x00, 0xff)
        # (status, position, force) = hand.wait_move_complete()

        while cont:
            time.sleep(0.05)
            # (status, position, force) = hand.wait_move_complete()
            # (position, force) = hand.get_instant_status()
            data = hand.get_instant_raw_status()
            # position_mm = hand.get_position_mm(position)
            # force_mA = hand.get_force_mA(force)
            # print('instant: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
            # print('data')
            # print(data)

            data2 = splitting(data.hex())
            data_decimal = hexbytearray2decbytearray(data)


            print("data2")
            print(data2)

            print("data_decimal")
            print(data_decimal)
            print("data")
            print(data)

            # TODO create a function "printing bytearray in hexadecimal"
            # TODO create a function "printing bytearray in decimal"
            # for this use all the funcitons already created on the top of this script
            # cause this is only useful for printing
            # for working, you should use this:

            print("data[6] == 255")
            print(data[6] == 255)

            print("data[6] == 0xff")
            print(data[6] == 0xff)



    except:
        print('Ctrl-c pressed')
        #TODO create handler to close the door

    hand.disconnect()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    test_robotiq()
