#!/usr/bin/env python3
import copy

import numpy as np
import numpy.random

a = np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [0, 2, 0],
              [0, 2, 0],
              [0, 2, 0],
              [0, 0, 3],
              [0, 0, 3],
              [0, 0, 3]])
b = copy.deepcopy(a)

print(a)
rng_state = numpy.random.get_state()
np.random.shuffle(a)

numpy.random.set_state(rng_state)
np.random.shuffle(b)

print(b)
print(a == b)

