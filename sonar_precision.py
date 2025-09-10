import numpy as np
import matplotlib.pyplot as plt

distances = np.array([30, 60, 90, 120, 150, 300])
sensor_values = np.array([307, 625, 953, 1251, 1573, 3106])
errors = sensor_values - distances
