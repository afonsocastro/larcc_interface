#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from larcc_classes.data_storage.DataForLearning import DataForLearning

# received_data = []

pulls = 0
pushes = 0
shakes = 0
twists = 0
start = 0
end = 0


def callback(data):
    global pulls
    global pushes
    global shakes
    global twists
    global start
    global end

    # received_data.append(data.data)
    i = data.data
    # for i in received_data:
    if i == "PULL":
        pulls += 1
    elif i == "PUSH":
        pushes += 1
    elif i == "SHAKE":
        shakes += 1
    elif i == "TWIST":
        twists += 1
    elif i == "START":
        start += 1
    elif i == "END":
        end += 1

    print("pulls: " + str(pulls))
    print("pushes: " + str(pushes))
    print("shakes: " + str(shakes))
    print("twists: " + str(twists))
    print("starts: " + str(start))
    print("ends: " + str(end))


if __name__ == '__main__':

    rospy.init_node('listener', anonymous=True)
    data_for_learning = DataForLearning()

    print(data_for_learning)
    # rospy.init_node('listener', anonymous=True)
    # rospy.Subscriber("ground_truth", String, callback)
    #
    #
    # rospy.spin()

