# -*- coding: utf-8 -*-

import socket
import time
import sys
import struct
import threading
from larcc_classes.gripper.GripperStatusDTO import GripperStatusDTO


# ------------------------------------------------------------------------------
# RobotiqHand class for python 2.7
# ------------------------------------------------------------------------------


def byte2bits(byte):
    new_format = "{0:08b}".format(byte)
    bits_list = [int(i) for i in new_format]
    return bits_list


class RobotiqHand:
    def __init__(self):
        self.so = None
        self._cont = False
        self._sem = None
        self._heartbeat_th = None
        self._max_position = 255
        self._min_position = 0
        # self.fingers_size = 85.0
        self.fingers_size = 140.0

    def _heartbeat_worker(self):
        while self._cont:
            self.status()
            time.sleep(0.5)

    def connect(self, ip, port):
        self.so = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.so.connect((ip, port))
        self._cont = True
        self._sem = threading.Semaphore(1)
        self._heartbeat_th = threading.Thread(target=self._heartbeat_worker)
        self._heartbeat_th.start()

    def disconnect(self):
        self._cont = False
        self._heartbeat_th.join()
        self._heartbeat_th = None
        self.so.close()
        self.so = None
        self._sem = None

    def _calc_crc(self, command):
        crc_registor = 0xFFFF
        for data_byte in command:
            tmp = crc_registor ^ data_byte
            for _ in range(8):
                if (tmp & 1 == 1):
                    tmp = tmp >> 1
                    tmp = 0xA001 ^ tmp
                else:
                    tmp = tmp >> 1
            crc_registor = tmp
        crc = bytearray(struct.pack('<H', crc_registor))
        return crc

    def send_command(self, command):
        with self._sem:
            crc = self._calc_crc(command)
            data = command + crc
            self.so.sendall(data)
            time.sleep(0.001)
            data = self.so.recv(1024)
        return bytearray(data)

    def status(self):
        command = bytearray(b'\x09\x03\x07\xD0\x00\x03')
        return self.send_command(command)

    def reset(self):
        command = bytearray(b'\x09\x10\x03\xE8\x00\x03\x06\x00\x00\x00\x00\x00\x00')
        return self.send_command(command)

    def activate(self):
        command = bytearray(b'\x09\x10\x03\xE8\x00\x03\x06\x01\x00\x00\x00\x00\x00')
        return self.send_command(command)

    # def close(self):
    #     command = bytearray(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\xFF\xFF\xFF\x42\x29')
    #     return self.send_command(command)

    def wait_activate_complete(self):
        while True:
            data = self.status()
            if data[5] != 0x00:
                return data[3]
            if data[3] == 0x31 and data[7] < 4:
                return data[3]

    def adjust(self):
        self.move(255, 64, 1)
        (status, position, force) = self.wait_move_complete()
        self._max_position = position
        self.move(0, 64, 1)
        (status, position, force) = self.wait_move_complete()
        self._min_position = position

    def get_position_mm(self, position):
        if position > self._max_position:
            position = self._max_position
        elif position < self._min_position:
            position = self._min_position
        position_mm = self.fingers_size * (self._max_position - position) / (self._max_position - self._min_position)
        # print 'max=%d, min=%d, pos=%d pos_mm=%.1f' % (self._max_position, self._min_position, position, position_mm)
        return position_mm

    def get_force_mA(self, force):
        return 10.0 * force

    # position: 0x00...open, 0xff...close
    # speed: 0x00...minimum, 0xff...maximum
    # force: 0x00...minimum, 0xff...maximum
    def move(self, position, speed, force):
        # print('move hand')
        command = bytearray(b'\x09\x10\x03\xE8\x00\x03\x06\x09\x00\x00\x00\x00\x00')
        command[10] = position
        command[11] = speed
        command[12] = force
        return self.send_command(command)

    # result: (status, position, force)
    def wait_move_complete(self):
        while True:
            data = self.status()
            if data[5] != 0x00:
                return -1, data[7], data[8]
            if data[3] == 0x79:
                return 2, data[7], data[8]
            if data[3] == 0xb9:
                return 1, data[7], data[8]
            if data[3] == 0xf9:
                return 0, data[7], data[8]

    # result: (position, force)
    def get_instant_status(self):
        data = self.status()
        return data[7], data[8]

    def get_instant_raw_status(self):
        data = self.status()
        return data

    def get_instant_gripper_status(self):
        data = self.status()

        bits0 = byte2bits(data[3])
        bits1 = byte2bits(data[4])
        bits2 = byte2bits(data[5])
        bits3 = byte2bits(data[6])
        bits4 = byte2bits(data[7])
        bits5 = byte2bits(data[8])

        # BYTE 0 | GRIPPER STATUS ----------------------------------------------------------------------------------
        # gACT
        is_activated = False
        if bits0[7] == 1:
            is_activated = True
        elif bits0[7] == 0:
            is_activated = False

        # gGTO
        going_to_position_request = False
        if bits0[4] == 1:
            going_to_position_request = True
        elif bits0[4] == 0:
            going_to_position_request = False

        # gSTA
        is_in_automatic_release_or_reset, activation_in_progress, activation_completed = False, False, False
        if bits0[2] == 0 and bits0[3] == 0:
            is_in_automatic_release_or_reset = True
        elif bits0[2] == 0 and bits0[3] == 1:
            activation_in_progress = True
        elif bits0[2] == 1 and bits0[3] == 1:
            activation_completed = True

        # gOBJ
        object_detected, fingers_in_motion_towards_requested_position, fingers_stopped_opening, fingers_stopped_closing,\
            fingers_are_at_requested_position = False, False, False, False, False
        if bits0[0] == 0 and bits0[1] == 0:
            fingers_in_motion_towards_requested_position = True
        elif bits0[0] == 0 and bits0[1] == 1:
            fingers_stopped_opening = True
            object_detected = True
        elif bits0[0] == 1 and bits0[1] == 0:
            fingers_stopped_closing = True
            object_detected = True
        elif bits0[0] == 1 and bits0[1] == 1:
            fingers_are_at_requested_position = True
        # ---------------------------------------------------------------------------------------------------

        # BYTE 1 ---------------------------------------------------------------------------------------------------
        # RESERVED
        # ---------------------------------------------------------------------------------------------------

        # BYTE 2 | FAULT STATUS ---------------------------------------------------------------------------------------
        # gFLT

        # Prior faults
        fault_action_delayed = False
        fault_missing_activation_bit = False

        # Minor faults
        fault_max_temperature_exceeded = False
        fault_no_communication_1_second = False

        # Major faults
        fault_under_minimum_voltage = False
        fault_automatic_release_in_progress = False
        fault_internal = False
        fault_activation = False
        fault_overcurrent_triggered = False
        fault_automatic_release_completed = False

        if data[5] == 5:
            fault_action_delayed = True
        elif data[5] == 7:
            fault_missing_activation_bit = True
        elif data[5] == 8:
            fault_max_temperature_exceeded = True
        elif data[5] == 9:
            fault_no_communication_1_second = True
        elif data[5] == 10:
            fault_under_minimum_voltage = True
        elif data[5] == 11:
            fault_automatic_release_in_progress = True
        elif data[5] == 12:
            fault_internal = True
        elif data[5] == 13:
            fault_activation =True
        elif data[5] == 14:
            fault_overcurrent_triggered = True
        elif data[5] == 15:
            fault_automatic_release_completed = True

        # ---------------------------------------------------------------------------------------------------

        # BYTE 3 | POSITION REQUEST ECHO -----------------------------------------------------------------------------
        # gPR
        requested_position = self.get_position_mm(data[6])
        # requested_position = data[6]
        # ---------------------------------------------------------------------------------------------------

        # BYTE 4 | POSITION ------------------------------------------------------------------------------------------
        # gPO
        actual_position = self.get_position_mm(data[7])
        # actual_position = data[7]
        # ---------------------------------------------------------------------------------------------------

        # BYTE 5 | CURRENT ----------------------------------------------------------------------------------------
        actual_force_motor_current = self.get_force_mA(data[8])
        # actual_force_motor_current = data[8]
        # ---------------------------------------------------------------------------------------------------

        gripper_status = GripperStatusDTO(
            is_activated=is_activated,
            going_to_position_request=going_to_position_request,
            activation_in_progress=activation_in_progress,
            activation_completed=activation_completed,
            is_in_automatic_release_or_reset=is_in_automatic_release_or_reset,
            object_detected=object_detected,
            fingers_in_motion_towards_requested_position=fingers_in_motion_towards_requested_position,
            fingers_stopped_opening=fingers_stopped_opening,
            fingers_stopped_closing=fingers_stopped_closing,
            fingers_are_at_requested_position=fingers_are_at_requested_position,
            fault_action_delayed=fault_action_delayed,
            fault_missing_activation_bit=fault_missing_activation_bit,
            fault_max_temperature_exceeded=fault_max_temperature_exceeded,
            fault_no_communication_1_second=fault_no_communication_1_second,
            fault_under_minimum_voltage=fault_under_minimum_voltage,
            fault_internal=fault_internal,
            fault_activation=fault_activation,
            fault_overcurrent_triggered=fault_overcurrent_triggered,
            fault_automatic_release_completed=fault_automatic_release_completed,
            fault_automatic_release_in_progress=fault_automatic_release_in_progress,
            requested_position=requested_position,
            actual_position=actual_position,
            actual_force_motor_current=actual_force_motor_current
        )
        return gripper_status
