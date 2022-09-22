#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from larcc_classes.gripper.GripperStatusDTO import GripperStatusDTO
if __name__ == '__main__':
    gripper_status = GripperStatusDTO(name= True, location= 42)
    print( gripper_status.name)
    print(gripper_status)
