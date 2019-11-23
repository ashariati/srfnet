import sys, os

import numpy as np
import cv2
import csv
import rosbag
import roslib
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from tf import transformations

def to_list(csvfile):
    data = []
    datareader = csv.reader(csvfile, delimiter=' ')
    for row in datareader:
        data.append(row)
    return data

def to_rostimes(data):
    timelist = []
    for t in data:
        timelist.append(rospy.rostime.Time.from_sec(float(t[0])))
    return timelist

def to_posemsgs(data):
    poselist = []
    data = np.array(data).astype(np.float)
    for p in data:
        G = np.vstack((p.reshape(3, 4), [0, 0, 0, 1]))
        posemsg = tmat_to_posemsg(G)
        poselist.append(posemsg)
    return poselist

def poses_to_simimu(poselist, timelist):
    
    imulist = [Imu()]
    for i, p1 in enumerate(poselist):

        # delta t
        dt = (timelist[i] - timelist[i-1]).to_sec()

        # frame to frame transformation
        G0 = posemsg_to_tmat(poselist[i-1])
        G1 = posemsg_to_tmat(p1)
        dG = np.dot(transformations.inverse_matrix(G0), G1)


    return imulist

def tmat_to_posemsg(G):
    q = transformations.quaternion_from_matrix(G)
    t = transformations.translation_from_matrix(G)
    posemsg = Pose()
    posemsg.position.x = t[0]
    posemsg.position.y = t[1]
    posemsg.position.z = t[2]
    posemsg.orientation.x = q[0]
    posemsg.orientation.y = q[1]
    posemsg.orientation.z = q[2]
    posemsg.orientation.w = q[3]
    return posemsg

def posemsg_to_tmat(msg):
    q = np.zeros((4,), dtype=np.float)
    q[0] = msg.orientation.x
    q[1] = msg.orientation.y
    q[2] = msg.orientation.z
    q[3] = msg.orientation.w
    t = np.zeros((3,), dtype=np.float)
    t[0] = msg.position.x
    t[1] = msg.position.y
    t[2] = msg.position.z
    R = transformations.quaternion_matrix(q)
    T = transformations.translation_matrix(t)
    G = np.dot(T, R)
    return G

datadir = sys.argv[1]
try:
    sys.argv[2]
except:
    seqs = range(1,21)
else:
    seqs = [int(sys.argv[2])]

destinationdir = os.path.join(datadir, '..', 'coarsesim')
if not os.path.exists(destinationdir):
    os.mkdir(destinationdir)

for i in iter(seqs):

    # find source
    sequencedir = os.path.join(datadir, 'sequences', str(i).zfill(2), 'image_0')

    # times 
    timesfile = open(os.path.join(datadir, 'sequences', str(i).zfill(2)) + '/times.txt')
    timelist = to_rostimes(to_list(timesfile))

    # ground truth poses
    posesfile = open(os.path.join(datadir, 'poses') + '/' + str(i).zfill(2) + '.txt')
    poselist = to_posemsgs(to_list(posesfile))

    # simulated imu from ground truth and time
    imulist = poses_to_simimu(poselist, timelist)
    

    # # calibration
    # calibfile = open(os.path.join(datadir, 'sequences', str(i).zfill(2)) + '/calib.txt')
    # calibreader = csv.reader(calibfile, delimiter=' ')
    # K0 = np.array((calibreader.next()[1:])).astype(np.float).reshape(3, 4)



