#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import signal
from RobotiqHand import RobotiqHand
from ast import literal_eval
import binascii

# ------------------------------------------------------------------------------
# test_robotiq.py
# ------------------------------------------------------------------------------
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


def hex_bytearray2dec_list(hex_bytearray):
    dec_list = []
    for i in hex_bytearray:
        dec_i = hex2dec(i)
        dec_list.append(dec_i)
    return dec_list


def byte2bits(byte):
    new_format = "{0:08b}".format(byte)
    bits_list = [int(i) for i in new_format]
    return bits_list


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

        # time.sleep(2)
        # print('open fast')
        # hand.move(0, 255, 0)
        # (status, position, force) = hand.wait_move_complete()

        time.sleep(1)
        print('close slow')
        hand.move(255, 0, 255)
        # (status, position, force) = hand.wait_move_complete()

        while cont:
            time.sleep(0.05)
            # (status, position, force) = hand.wait_move_complete()
            # (position, force) = hand.get_instant_status()
            # data = hand.get_instant_raw_status()
            status = hand.get_instant_gripper_status()
            # position_mm = hand.get_position_mm(position)
            # force_mA = hand.get_force_mA(force)
            # print('instant: position = {:.1f}mm, force = {:.1f}mA '.format(position_mm, force_mA))
            # print('data')
            # print(data)
            print(status)
            # data2 = splitting(data.hex())
            # data_decimal = hex_bytearray2dec_list(data)

            # print("data2")
            # print(data2)
            #
            # print("data_decimal")
            # print(data_decimal)
            # print("data")
            # print(data)

            # TODO create a function "printing bytearray in hexadecimal"
            # TODO create a function "printing bytearray in decimal"
            # for this use all the funcitons already created on the top of this script
            # cause this is only useful for printing
            # for working, you should use this:

            # print("data[7]")
            # print(data[7])
            #
            # bits = byte2bits(data[7])
            # print("bits of data[7] byte")
            # print(bits)
            # print("bits[0]")
            # print(bits[0])
            # print("bits[7]")
            # print(bits[7])


            # print("data[6] == 0xff")
            # print(data[6] == 0xff)



    except:
        print('Ctrl-c pressed')
        # TODO create handler to close the door

    hand.disconnect()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handler)
    test_robotiq()