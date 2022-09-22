#!/usr/bin/env python3

import time
from larcc_classes.gripper.RobotiqHand import RobotiqHand

HOST = "192.168.56.2"
PORT = 54321

hand = RobotiqHand()

hand.connect(HOST, PORT)
print("Connected")

hand.move(0, 255, 0)

hand.wait_move_complete()

hand.disconnect()
print("Disconnected")
