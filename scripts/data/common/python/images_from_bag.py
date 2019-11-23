import sys, os

import rospy
import rosbag
import cv2
import numpy as np

from tf import transformations
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if len(sys.argv) < 2:
    print('No image topic given')
    sys.exit()
imagetopic = sys.argv[1]

if len(sys.argv) < 3:
    print('No bagfile given')
    sys.exit()


for bagfile in sys.argv[2:]:

    print('Processing ' + os.path.basename(bagfile) + ' ...')

    # initialize image directory
    sequencedir = os.path.join(os.path.dirname(bagfile), 'raw_sequences')
    if not os.path.isdir(sequencedir):
        os.mkdir(sequencedir)
    datadir = os.path.join(sequencedir, os.path.basename(bagfile).split('.')[0])
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    imagedir = os.path.join(datadir, 'images')
    if not os.path.isdir(imagedir):
        os.mkdir(imagedir)

    # bag data
    count = 0
    timesmat = []

    # parse bag
    bridge = CvBridge()
    bag = rosbag.Bag(bagfile)
    nimages = bag.get_message_count(imagetopic)
    for topic, msg, t in bag.read_messages(topics=[imagetopic]):

        image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        imagefile = os.path.join(imagedir, str(count).zfill(5) + '.png')
        cv2.imwrite(imagefile, image, params=[int(cv2.IMWRITE_PNG_COMPRESSION)])

        time = msg.header.stamp.to_sec()
        timesmat.append([time])

        print('Writing image {0} / {1} ...'.format(count+1, nimages))
        count += 1

    timesmat = np.array(timesmat)
    np.savetxt(os.path.join(datadir, 'times.txt'), timesmat, fmt='%.9f')

    bag.close()
