import sys, os

import rospy
import rosbag
import numpy as np

from tf import transformations
from geometry_msgs.msg import TransformStamped

if len(sys.argv) < 2:
    print('No pose topic given')
    sys.exit()
posetopic = sys.argv[1]

if len(sys.argv) < 3:
    print('No bagfile given')
    sys.exit()


for bagfile in sys.argv[2:]:

    print('Processing ' + os.path.basename(bagfile) + ' ...')

    # initialize data directory
    posesdir = os.path.join(os.path.dirname(bagfile), 'raw_poses')
    if not os.path.isdir(posesdir):
        os.mkdir(posesdir)
    posefile = os.path.basename(bagfile).split('.')[0] + '.txt'

    # bag data
    count = 0
    posedata = []

    # parse bag
    bag = rosbag.Bag(bagfile)
    nmsgs = bag.get_message_count(posetopic)
    for topic, msg, t in bag.read_messages(topics=[posetopic]):

        time = msg.header.stamp.to_sec()

        t = np.array([
            msg.transform.translation.x, 
            msg.transform.translation.y, 
            msg.transform.translation.z])
        q = np.array([
            msg.transform.rotation.x, 
            msg.transform.rotation.y, 
            msg.transform.rotation.z, 
            msg.transform.rotation.w])

        Gt = transformations.translation_matrix(t)
        Gr = transformations.quaternion_matrix(q)
        G = np.dot(Gr, Gt)

        row = [time] + G.flatten().tolist()
        posedata.append(row)

        print('Reading message {0} / {1} ...'.format(count+1, nmsgs))
        count += 1

    posedata = np.array(posedata)
    np.savetxt(os.path.join(posesdir, posefile), posedata, fmt='%.12f')

    bag.close()