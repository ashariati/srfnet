import sys, os

import rospy
import rosbag
import numpy as np

from sensor_msgs.msg import Imu

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
    imudir = os.path.join(os.path.dirname(bagfile), 'raw_imu')
    if not os.path.isdir(imudir):
        os.mkdir(imudir)
    imufile = os.path.basename(bagfile).split('.')[0] + '.txt'

    # bag data
    count = 0
    imudata = []

    # parse bag
    bag = rosbag.Bag(bagfile)
    nmsgs = bag.get_message_count(posetopic)
    for topic, msg, t in bag.read_messages(topics=[posetopic]):

        time = msg.header.stamp.to_sec()

        omega = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        a = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        row = [time] + omega.tolist() + a.tolist()
        imudata.append(row)

        print('Reading message {0} / {1} ...'.format(count+1, nmsgs))
        count += 1

    imudata = np.array(imudata)
    np.savetxt(os.path.join(imudir, imufile), imudata, fmt='%.12f')

    bag.close()