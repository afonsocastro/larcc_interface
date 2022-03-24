#! /usr/bin/python3

from MoveGroupPythonInterface import MoveGroupPythonInterface
import rospy
from std_msgs.msg import String
import json

# --------------------------------------------------------------------
# -------------------------Initialization-----------------------------
# --------------------------------------------------------------------
arm = MoveGroupPythonInterface()


def request_arm_callback(data):
    pub = rospy.Publisher('arm_response', String, queue_size=10)

    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("I heard %s", data.data)

    arm_request_dict = json.loads(data.data)

    if arm_request_dict['action'] == 'move_to_initial_pose':
        state = move_arm_to_initial_pose()
        if state == True:
            result = 'Arm is now at initial pose.'
            print(result)
            pub.publish(result)

    elif arm_request_dict['action'] == 'move_to_pose_goal':
        trans = arm_request_dict['trans']
        quat = arm_request_dict['quat']

        state = move_arm_to_pose_goal(trans, quat)
        if state == True:
            result = 'Arm is now at requested pose goal.'
            print(result)
            pub.publish(result)

    elif arm_request_dict['action'] == 'move_to_joints_state':
        joints_state = arm_request_dict['joints']

        state = move_arm_to_joints_state(joints_state)
        if state == True:
            result = 'Arm is now at requested joints state goal.'
            print(result)
            pub.publish(result)


def move_arm_to_initial_pose():
    state = arm.go_to_joint_state(0.01725006103515625, -1.9415461025633753, 1.8129728476153772, -1.5927173099913539, -1.5878670851336878, 0.03150486946105957)
    return state


def move_arm_to_pose_goal(trans, quat):
    state = arm.go_to_pose_goal(trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3])
    return state


def move_arm_to_joints_state(joints_state):
    state = arm.go_to_joint_state(joints_state[0], joints_state[1], joints_state[2], joints_state[3], joints_state[4], joints_state[5])
    return state


if __name__ == '__main__':
    move_group = arm.move_group

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('arm_controller', anonymous=True)

    rospy.Subscriber("arm_request", String, request_arm_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

