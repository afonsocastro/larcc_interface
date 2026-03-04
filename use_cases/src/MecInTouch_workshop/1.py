#!/usr/bin/env python3

import rospy  # Import ROS Python library
from UR10eArm import UR10eArm
# from RobotiqHand import RobotiqHand


if __name__ == '__main__':

    # Initialization------------------------------------------------
    rospy.init_node('my_node', anonymous=True)  # Initialize the node
    rate = rospy.Rate(1)
    manipulator = UR10eArm()

    # Para usar o gripper, descomentar:
    # hand = RobotiqHand()
    # hand.connect("192.168.56.2", 54321)
    # hand.reset()
    # hand.activate()
    # --------------------------------------------------------------

    #  Your code should start from here:

    current_pose = manipulator.move_group.get_current_pose().pose
    print("\nActual TCP Pose: ")
    print(current_pose)

    rospy.sleep(2) #Comando para "dormir" 2 segundos. Pode ser útil se quiserem fazer um compasso de espera entre alguma tarefa."

    current_joints = manipulator.move_group.get_current_joint_values()
    print("\nActual Joint Values:")
    print(current_joints)

    # hand.move(255, 255, 1) # fechar gripper à max velocidade

    manipulator.go_to_joint_state(0.69, -0.91, 0.84, -1.91, 1.41, -0.08, 0.2, 0.2)
    manipulator.go_to_joint_state(0.50, -0.88, 0.10, -1.21, 1.2, -0.54, 0.78, 0.78)
    manipulator.go_to_joint_state(0.70, -1.08, 0.78, -1.21, 1.8, -0.74, 0.78, 0.88)


    print("\nCheguei!! :D\n")

    # hand.move(0, 255, 1) # abrir gripper à max velocidade

    current_joints = manipulator.move_group.get_current_joint_values()
    print("\nNew Joint Values:")
    for n in range(0,6):
        print(current_joints[n])