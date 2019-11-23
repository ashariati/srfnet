import sys
import os

import numpy as np

import pprint
import pdb

model = sys.argv[1]
offset = sys.argv[2]

data = []
for sequence in sys.argv[3:]:

    modeldir = os.path.join(os.path.abspath(sequence),
            'offset_%s' % offset,
            model)

    data.extend(np.loadtxt(os.path.join(modeldir, 'summary_data.txt'), delimiter=' '))

data = np.array(data)
epes = data[:, 0]
pcents = data[:, 1]
thetas = data[:, 2]

print("Avg. Endpoint Error = %.3f" % np.mean(epes))
print("Avg. Percent EPE = %.2f" % np.mean(pcents))
print("Avg. Angular Difference = %.2f" % np.mean(thetas))

