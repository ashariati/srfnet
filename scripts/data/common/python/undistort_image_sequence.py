import sys, os

import cv2
import numpy as np
import yaml
import glob
import shutil

def load_camera_calibration(f):

    with open(f, 'r') as stream:

        try:
            calib = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            sys.exit()

    return calib

if len(sys.argv) < 2:
    print('No calibration file given')
    sys.exit()
calibfile = sys.argv[1]

if len(sys.argv) < 3:
    print('No image directory given given')
    sys.exit()

# read calibration matrix
calibration = load_camera_calibration(calibfile)
K = np.array(calibration['camera_matrix']['data']).reshape((3, 3))
distortion = np.array(calibration['distortion_coefficients']['data'])

# for each sequence directory
for inseqdir in sys.argv[2:]:

    print('Processing ' + inseqdir + ' ...')

    # make destination sequences directory
    outseqdir = os.path.join(os.path.dirname(inseqdir), '..', 'sequences')
    if not os.path.isdir(outseqdir):
        os.mkdir(outseqdir)
    outdestdir = os.path.join(outseqdir, os.path.basename(inseqdir))
    if not os.path.isdir(outdestdir):
        os.mkdir(outdestdir)
    outimagedir = os.path.join(outdestdir, 'images')
    if not os.path.isdir(outimagedir):
        os.mkdir(outimagedir)

    # write undistorted calibration
    np.savetxt(outdestdir + '/calib.txt', K, fmt='%.9f')

    # copy times file
    shutil.copyfile(inseqdir + '/times.txt', outdestdir + '/times.txt')

    # for each image
    inimagedir = os.path.join(inseqdir, 'images')
    for imagefile in glob.glob(inimagedir + '/*.png'):

        distorted_image = cv2.imread(imagefile)
        undistorted_image = cv2.undistort(distorted_image, K, distortion)
        cv2.imwrite(outimagedir + '/' + os.path.basename(imagefile), undistorted_image)