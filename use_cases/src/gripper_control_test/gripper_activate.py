#!/usr/bin/env python3
from gripper.src.RobotiqHand import RobotiqHand

HOST = "192.168.56.2"
PORT = 54321

hand = RobotiqHand()

hand.connect(HOST, PORT)
print("Connected")

hand.reset()
hand.activate()

hand.disconnect()
print("Disconnected")
