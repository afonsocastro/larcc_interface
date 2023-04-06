#!/usr/bin/env python3

from scipy.interpolate import InterpolatedUnivariateSpline
from numpy import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([1, 1, 2, 1, -2])
    f = InterpolatedUnivariateSpline(x, y, k=1)  # k=1 gives linear interpolation
    print("f.integral(1.5, 2.2)")
    print(f.integral(1.5, 2.2))
