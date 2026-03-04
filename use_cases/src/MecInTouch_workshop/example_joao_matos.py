#!/usr/bin/env python3

import rospy  # Import ROS Python library
from UR10eArm import UR10eArm
# from RobotiqHand import RobotiqHand


# Funções disponíveis:

# Braço robótico:----------------------------------------------------------

# 1) manipulator.get_current_pose().pose:
# retorna o valor da posição (x,y,z) e da rotação (x,y,z,w) do tool0 em relação ao world
#
# 2) manipulator.get_current_joint_values():
# retorna o valor das 6 juntas do manipulador (em radianos)
#
# 3) manipulator.go_to_joint_state(j1, j2, j3, j4, j5, j6, vel, a):
# comanda o robô para se deslocar até ao destino definido das 6 juntas (j1,j2,j3,j4,j5,j6) em radianos.
# É possível escolher a velocidade e aceleração (vel, a) com que o robô fará esse percurso (entre 0 e 1).
# [ shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]

# 4) manipulator.go_to_pose_goal(trans_x, trans_y, trans_z, q1, q2, q3, q4, vel, a):
# comanda o robô para se deslocar até ao destino definido pelas coordenadas do tool0 (x,y,z,q1,q2,q3,q4).
# É possível escolher a velocidade e aceleração (vel, a) com que o robô fará esse percurso (entre 0 e 1).


# Gripper: --------------------------------------------

# 1) hand.move(pos, speed, 1):
# função para fechar/abrir a garra. pos [0,255] (0=aberta; 255=fechada). speed [0,255] (0= lento; 255=rápido)
#
# 2) (status, position, force) = hand.wait_move_complete():
# função para aguardar que o movimento do gripper termine para que o programa continue.
# Se não for colocado, o programa continua enquanto o gripper ainda está a mexer.


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

    manipulator.go_to_joint_state(0, 0, 0, 0, 0, 0, 0.2, 0.2)
    rospy.sleep(2)
    print("\nPosição inicial\n")
    manipulator.go_to_joint_state(0.69, -0.91, 0.84, -1.91, 1.41, -0.08, 0.2, 0.2)
    rospy.sleep(2)
    print("\nCheguei!! :D\n")
    manipulator.go_to_joint_state(0.69, -1.41, 0.84, -1.91, 1.41, -0.08, 0.2, 0.2)
    rospy.sleep(2)
    print("\nCheguei!! :D\n")
    manipulator.go_to_joint_state(0, 0, 0, 0, 0, 0, 0.2, 0.2)

    # hand.move(0, 255, 1) # abrir gripper à max velocidade

    current_joints = manipulator.move_group.get_current_joint_values()
    print("\nNew Joint Values:")
    for n in range(0,6):
        print(current_joints[n])

    # hand.disconnect()
