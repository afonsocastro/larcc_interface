#!/usr/bin/env python3

from larcc_classes.gripper.RobotiqHand import RobotiqHand

HOST = "192.168.56.2"
PORT = 54321

hand = RobotiqHand()

hand.connect(HOST, PORT)
print("Connected")

hand.reset()
hand.activate()
result = hand.wait_activate_complete()

hand.adjust()
hand.disconnect()
print("Disconnected")
