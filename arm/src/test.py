#! /usr/bin/python3

from MoveGroupPythonInterface import MoveGroupPythonInterface
import rospy

if __name__ == '__main__':
    rospy.init_node("move_group_python_interface", anonymous=True)
    # --------------------------------------------------------------------
    # -------------------------initialization-----------------------------
    # --------------------------------------------------------------------
    tutorial = MoveGroupPythonInterface()
    move_group = tutorial.move_group

    current_pose = move_group.get_current_pose().pose
    print("current_pose")
    print(current_pose)

    current_joints = move_group.get_current_joint_values()
    print("current_joints")
    print(current_joints)
