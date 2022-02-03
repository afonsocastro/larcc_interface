#!/usr/bin/env python

# ------------------------
#    IMPORT MODULES      #
# ------------------------
import copy

import numpy as np
import cv2
import tf
from interactive_markers.menu_handler import *
from visualization_msgs.msg import *
from tf.listener import TransformListener


# ------------------------
#      BASE CLASSES      #
# ------------------------

class TransformationT():
    """ Base class for geometric transformation
    """

    def __init__(self, parent_frame_id, frame_id):
        self.parent_frame_id = parent_frame_id
        self.frame_id = frame_id
        self.matrix = np.identity(4, dtype=np.float)

        self.stamp = rospy.Time.now()

    def __str__(self):
        return 'Transform from ' + str(self.parent_frame_id) + ' to ' + str(self.frame_id) + ' at time ' + str(
            self.stamp) + '\n' + str(self.matrix)

    def getTranslation(self, homogeneous=False):
        if homogeneous:
            return self.matrix[0:4, 3]
        else:
            return self.matrix[0:3, 3]

    def setTranslation(self, value):
        self.matrix[0:3, 3] = value

    def setTranslationFromPosePosition(self, trans):
        self.matrix[0, 3] = trans.x
        self.matrix[1, 3] = trans.y
        self.matrix[2, 3] = trans.z

    def getRotation(self):
        return self.matrix[0:3, 0:3]

    def setRotation(self, matrix):
        self.matrix[0:3, 0:3] = matrix

    def setQuaternion(self, q):
        matrix = copy.deepcopy(tf.transformations.quaternion_matrix(q))
        self.matrix[0:3, 0:3] = matrix[0:3, 0:3]

    def setQuaternionFromPoseQuaternion(self, pose_q):
        q = (pose_q.x, pose_q.y, pose_q.z, pose_q.w)
        matrix = copy.deepcopy(tf.transformations.quaternion_matrix(q))
        self.matrix[0:3, 0:3] = matrix[0:3, 0:3]

    def getQuaternion(self):
        m = copy.deepcopy(self.matrix)
        m[0, 3] = 0
        m[1, 3] = 0
        m[2, 3] = 0
        return tf.transformations.quaternion_from_matrix(m)

    def getEulerAngles(self):
        return tf.transformations.euler_from_matrix(self.matrix)

    def setRodrigues(self, matrix):
        self.setRotation(self.rodriguesToMatrix(matrix))

    def getRodrigues(self):
        return self.matrixToRodrigues(self.getRotation())

    def matrixToRodrigues(self, matrix):
        rods, _ = cv2.Rodrigues(matrix[0:3, 0:3])
        rods = rods.transpose()
        rodrigues = rods[0]
        return rodrigues

    def rodriguesToMatrix(self, r):
        rod = np.array(r, dtype=np.float)
        matrix = cv2.Rodrigues(rod)
        return matrix[0]


class MarkerPoseC:
    def __init__(self, position, orientation, frame_id, child_frame_id):
        self.position = position
        self.orientation = orientation
        self.frame_id = frame_id
        self.child_frame_id = child_frame_id

    def __str__(self):
        return str(self.position) + "\n" + str(self.orientation)

    __repr__ = __str__
