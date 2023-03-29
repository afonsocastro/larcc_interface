#! /usr/bin/python3

from arm.msg import MoveArmAction
from larcc_classes.arm.UR10eArm import UR10eArm
import rospy
from std_msgs.msg import String


# --------------------------------------------------------------------
# -------------------------Initialization-----------------------------
# --------------------------------------------------------------------
arm = UR10eArm()


def request_arm_callback(msg):
    # rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("I heard %s", msg)

    if msg.action == 'move_to_initial_pose':
        assert 0 < msg.velocity <= 1 and 0 < msg.acceleration <= 1

        state = move_arm_to_initial_pose(msg.velocity, msg.acceleration)
        if state == True:
            result = 'Arm is now at initial pose.'
            print(result)
            pub.publish(result)

    elif msg.action == 'move_to_pose_goal':
        assert len(msg.goal) == 7 and 0 < msg.velocity <= 1 and 0 < msg.acceleration <= 1
        trans = msg.goal[0:3]
        quat = msg.goal[3:7]

        state = move_arm_to_pose_goal(trans, quat, msg.velocity, msg.acceleration)
        if state == True:
            result = 'Arm is now at requested pose goal.'
            print(result)
            pub.publish(result)

    elif msg.action == 'move_to_joints_state':
        assert len(msg.goal) == 6 and 0 < msg.velocity <= 1 and 0 < msg.acceleration <= 1
        joints_state = msg.goal

        state = move_arm_to_joints_state(joints_state, msg.velocity, msg.acceleration)
        if state == True:
            result = 'Arm is now at requested joints state goal.'
            print(result)
            pub.publish(result)


def move_arm_to_initial_pose(vel, a):
    state = arm.go_to_joint_state(0.01725006103515625, -1.9415461025633753, 1.8129728476153772, -1.5927173099913539, -1.5878670851336878, 0.03150486946105957, vel, a)
    return state


def move_arm_to_pose_goal(trans, quat, vel, a):
    state = arm.go_to_pose_goal(trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3], vel, a)
    return state


def move_arm_to_joints_state(joints_state, vel, a):
    state = arm.go_to_joint_state(joints_state[0], joints_state[1], joints_state[2], joints_state[3], joints_state[4], joints_state[5], vel, a)
    return state


if __name__ == '__main__':
    move_group = arm.move_group

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('arm_controller', anonymous=True)

    pub = rospy.Publisher('arm_response', String, queue_size=10)
    rospy.Subscriber("arm_request", MoveArmAction, request_arm_callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

