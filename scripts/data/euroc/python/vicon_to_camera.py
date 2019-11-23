import sys, os

import numpy as np
import yaml

def load_calibration(f):

    with open(f, 'r') as stream:
        try:
            calibration = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
            sys.exit()
    return calibration

if len(sys.argv) < 2:
    print('No body-vicon calibration file given')
    sys.exit()
bvcalibfile = sys.argv[1]

if len(sys.argv) < 3:
    print('No body-camera calibration file given')
    sys.exit()
bccalibfile = sys.argv[2]

if len(sys.argv) < 4:
    print('No pose data given')
    sys.exit()


# vicon calibration
bvcalib = load_calibration(bvcalibfile)
T_bv = np.array(bvcalib['T_BS']['data']).reshape((4,4))
T_vb = np.linalg.inv(T_bv)

# camera calibration
bccalib = load_calibration(bccalibfile)
T_bc = np.array(bccalib['T_BS']['data']).reshape((4,4))

# compose for vicon-body transform
T_vc = np.dot(T_vb, T_bc)

for posefile in sys.argv[3:]:

    print('Processing ' + os.path.basename(posefile) + ' ...')

    # initialize data directory
    posesdir = os.path.join(os.path.dirname(posefile), '..', 'poses')
    if not os.path.isdir(posesdir):
        os.mkdir(posesdir)
    tfposefile = os.path.basename(posefile)

    data = np.loadtxt(posefile)
    time = data[:, 0]
    T_wv = data[:, 1:].T

    # matrix multiplication as vector multiplication
    M = np.kron(np.eye(4), T_vc.T)

    # transform poses to camera frame
    T_wc = np.dot(M, T_wv)

    # write out new poses
    np.savetxt(os.path.join(posesdir, tfposefile), np.vstack((time, T_wc)).T, fmt='%.9f')
    
