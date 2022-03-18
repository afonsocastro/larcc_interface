#!/usr/bin/env python3

from time import sleep
import rospy
from std_msgs.msg import String
import json
import tf
from transformation_t import *
import time


def gripper_response_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("I heard %s", data.data)


def arm_response_callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    print("I heard %s", data.data)


if __name__ == '__main__':
    rospy.init_node('demo1', anonymous=True)

    rospy.Subscriber("gripper_response", String, gripper_response_callback)
    pub_gripper = rospy.Publisher('gripper_request', String, queue_size=10)

    rospy.Subscriber("arm_response", String, arm_response_callback)
    pub_arm = rospy.Publisher('arm_request', String, queue_size=10)

    rate = rospy.Rate(10.0)  # 10hz

    # -----------------ARM INITIAL POSE------------------------------
    sleep(10)
    arm_dict = {'action': 'move_to_initial_pose'}
    encoded_data_string = json.dumps(arm_dict)
    pub_arm.publish(encoded_data_string)
    # ----------------------------------------------------------------------

    # -----------------GRIPPER ACTIVATION------------------------------
    sleep(5)
    my_dict = {'action': 'init'}
    encoded_data_string = json.dumps(my_dict)
    rospy.loginfo(encoded_data_string)
    pub_gripper.publish(encoded_data_string)
    # ----------------------------------------------------------------------

    listener = tf.TransformListener()

    br = tf.TransformBroadcaster()
    wood_block_T_tool0_controller = TransformationT("wood_block", "tool0_controller")
    z_offset = 0.3

    # y_offset = 0.04
    y_offset = 0.0

    x_offset = -0.03
    # x_offset = 0.005
    fixed_trans = [0.03 + x_offset, -0.02 + y_offset, 0.15 + z_offset]

    wood_block_T_tool0_controller.setTranslation(fixed_trans)

    fixed_rot = [[-1.0000000, 0.0000000, 0.0000000],
                 [0.0000000, 1.0000000, 0.0000000],
                 [0.0000000, 0.0000000, -1.0000000]]

    wood_block_T_tool0_controller.setRotation(fixed_rot)
    start_time = time.time()
    first_warning = 0
    while not rospy.is_shutdown():
        sleep(0.5)
        try:
            (trans, rot) = listener.lookupTransform('/wood_block', '/base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("WAS NOT ABLE TO GET wood_block to base_link TF")
            continue

        base_link_T_wood_block = TransformationT("base_link", "wood_block")
        base_link_T_wood_block.setTranslation(trans)
        base_link_T_wood_block.setQuaternion(rot)

        base_link_T_tool0_controller = TransformationT("base_link", "tool0_controller")

        base_link_T_tool0_controller.matrix = np.dot(wood_block_T_tool0_controller.matrix,
                                                     base_link_T_wood_block.matrix)

        base_link_T_goal = TransformationT("goal2", "base_link")
        base_link_T_goal.matrix = np.linalg.inv(base_link_T_tool0_controller.matrix)
        goal_trans = base_link_T_goal.getTranslation()
        goal_quat = base_link_T_goal.getQuaternion()

        # -----------------ARM MOVE TO WOOD-BLOCK------------------------------
        # sleep(5)
        arm_dict = {'action': 'move_to_pose_goal', 'trans': [goal_trans[0], goal_trans[1], goal_trans[2]],
                    'quat': [goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3]]}
        encoded_data_string = json.dumps(arm_dict)
        pub_arm.publish(encoded_data_string)
        # ----------------------------------------------------------------------

        now_time = time.time()

        time_elapsed = (now_time - start_time)

        if time_elapsed > 30 and first_warning == 0:
            first_warning = 1
            # -----------------GRIPPER CLOSING FAST------------------------------
            # sleep(5)
            # values = [position, speed, force]
            my_dict = {'action': 'move', 'values': [255, 255, 0]}
            encoded_data_string = json.dumps(my_dict)
            rospy.loginfo(encoded_data_string)
            pub_gripper.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------GRIPPER OPPENING FAST------------------------------
            # sleep(5)
            # values = [position, speed, force]
            my_dict = {'action': 'move', 'values': [0, 255, 0]}
            encoded_data_string = json.dumps(my_dict)
            rospy.loginfo(encoded_data_string)
            pub_gripper.publish(encoded_data_string)
            # ----------------------------------------------------------------------

        if time_elapsed > 35 and first_warning == 1:
            first_warning = 0
            # -----------------ARM MOVE DOWN TO WOOD-BLOCK------------------------------
            # sleep(5)
            arm_dict = {'action': 'move_to_pose_goal', 'trans': [goal_trans[0], goal_trans[1], goal_trans[2]-0.1],
                        'quat': [goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3]]}
            encoded_data_string = json.dumps(arm_dict)
            pub_arm.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------GRIPPER CLOSING FAST------------------------------
            sleep(1)
            # values = [position, speed, force]
            my_dict = {'action': 'move', 'values': [255, 255, 0]}
            encoded_data_string = json.dumps(my_dict)
            rospy.loginfo(encoded_data_string)
            pub_gripper.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------ARM MOVE UP------------------------------
            sleep(1)
            arm_dict = {'action': 'move_to_pose_goal', 'trans': [goal_trans[0], goal_trans[1], goal_trans[2]+0.2],
                        'quat': [goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3]]}
            encoded_data_string = json.dumps(arm_dict)
            pub_arm.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------ARM MOVE ABOVE THE FINAL BOX------------------------------
            sleep(1)
            arm_dict = {'action': 'move_to_joints_state',
                        'joints': [0.33516550064086914, -1.163656548862793, 1.3741701284991663, -1.8040539226927699,
                                   -1.5384181181537073, 0.3482170104980469]}
            encoded_data_string = json.dumps(arm_dict)
            pub_arm.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------ARM MOVE TO FINAL BOX (JUST DOWN)------------------------------
            sleep(2)
            arm_dict = {'action': 'move_to_joints_state',
                        'joints': [0.34529924392700195, -1.0220564168742676, 1.5045183340655726, -2.004709859887594,
                                   -1.5726912657367151, 0.3451671600341797]}
            encoded_data_string = json.dumps(arm_dict)
            pub_arm.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------GRIPPER OPPENING FAST------------------------------
            sleep(3)
            # values = [position, speed, force]
            my_dict = {'action': 'move', 'values': [0, 255, 0]}
            encoded_data_string = json.dumps(my_dict)
            rospy.loginfo(encoded_data_string)
            pub_gripper.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------ARM MOVE ABOVE THE FINAL BOX------------------------------
            sleep(1)
            arm_dict = {'action': 'move_to_joints_state',
                        'joints': [0.33516550064086914, -1.163656548862793, 1.3741701284991663, -1.8040539226927699,
                                   -1.5384181181537073, 0.3482170104980469]}
            encoded_data_string = json.dumps(arm_dict)
            pub_arm.publish(encoded_data_string)
            # ----------------------------------------------------------------------

            # -----------------ARM INITIAL POSE------------------------------
            sleep(1)
            arm_dict = {'action': 'move_to_initial_pose'}
            encoded_data_string = json.dumps(arm_dict)
            pub_arm.publish(encoded_data_string)
            # ----------------------------------------------------------------------

        rate.sleep()

    # # -----------------GRIPPER ACTIVATION------------------------------
    # sleep(5)
    # my_dict = {'action': 'init'}
    # encoded_data_string = json.dumps(my_dict)
    # rospy.loginfo(encoded_data_string)
    # pub_gripper.publish(encoded_data_string)
    # # ----------------------------------------------------------------------
    #
    # # -----------------GRIPPER CLOSING FAST------------------------------
    # sleep(5)
    # # values = [position, speed, force]
    # my_dict = {'action': 'move', 'values': [255, 255, 0]}
    # encoded_data_string = json.dumps(my_dict)
    # rospy.loginfo(encoded_data_string)
    # pub_gripper.publish(encoded_data_string)
    # # ----------------------------------------------------------------------
    #
    # # -----------------GRIPPER DE-ACTIVATION------------------------------
    # sleep(5)
    # my_dict = {'action': 'disconnect'}
    # encoded_data_string = json.dumps(my_dict)
    # rospy.loginfo(encoded_data_string)
    # pub_gripper.publish(encoded_data_string)
    # # ----------------------------------------------------------------------


    rospy.spin()
