import numpy as np
import matplotlib.pyplot as plt

cm60 = np.array([624, 623, 623, 624, 623])
cm120 = np.array([1246, 1246, 1246, 1250, 1250])
cm180 = np.array([1873, 1872, 1872, 1872, 1873])
cm240 = np.array([2516, 2516, 2512, 2514, 2514])
cm300 = np.array([3107, 3106, 3106, 3106, 3106])

measurements = [cm60, cm120, cm180, cm240, cm300]
distances = np.array([600, 1200, 1800, 2400, 3000])

means = np.array([np.mean(m) for m in measurements])

errors = [m - d for m, d in zip(measurements, distances)]
std = np.array([np.std(e, ddof=1) for e in errors])

for s, d in zip(std, distances):
    print(f"std at {int(d/10)} cm: {s} mm")

plot = plt.plot(distances, means)
plt.xlabel("Distance (mm)")
plt.ylabel("mean of measurement (mm)")
plt.grid(True)
plt.show()
