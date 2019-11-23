import sys, os

import csv
import cv2
import glob
import numpy as np

def visualizerProgram(nimages):

    # initialize states
    statemap = np.zeros((2, nimages), dtype=np.int)
    statemap[0, :] = np.arange(0, nimages)
    initstate = (0, 0)

    # define transition function
    T = {}
    for j in range(0, nimages):

        # add state key
        T[(0, j)] = {}

        # left arrow
        T[(0, j)][81] = (0, j-1 if (j-1) >= 0 else 0)

        # right arrow
        T[(0, j)][83] = (0, j+1 if (j+1) < nimages else j)

        # down arrow
        T[(0, j)][84] = (1, j)

    for j in range(0, nimages):

        # add state key
        T[(1, j)] = {}

        # up arrow
        T[(1, j)][82] = (0, j)

    # create program
    def prog(state, keycmd):

        try:
            state = T[state][keycmd]
        except KeyError:
            state = state

        return state

    return prog, statemap, initstate

if len(sys.argv) < 2:
    print('No directory provided')
    sys.exit()

# find images
dirname = sys.argv[1]
imagefiles = glob.glob(dirname  + '/*.png')
imagefiles.sort()
nimages = len(imagefiles)

# load images
images = []
for f in imagefiles:
    images.append(cv2.imread(f))

# parse scale from dirname
scale = int(dirname.split('/')[-2][:-1])

# initialize state
viewerId = 0
prog, statemap, state = visualizerProgram(nimages)

# visualize
print('Press Left/Right arrows to compare sequential images')
print('Press Up/Down arrows to compare current image to first image')
print('Press ESC to exit')
key = None
imageid = 0
while key != 27:

    image = images[imageid]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    cv2.imshow('image', image)
    key = cv2.waitKey(0)

    state = prog(state, key)
    imageid = statemap[state[0], state[1]]

cv2.destroyAllWindows()