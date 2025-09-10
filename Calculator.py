import math
import matplotlib.pyplot as plt


data = {
    600:  [624, 623, 623, 624, 623],
    1200: [1246, 1246, 1246, 1250, 1250],
    1800: [1873, 1872, 1872, 1872, 1873],
    2400: [2516, 2516, 2512, 2514, 2514],
    3000: [3107, 3106, 3106, 3106, 3106]
}

data_cm = {x/10: [xi/10 for xi in obs] for x, obs in data.items()}


actual = []
measured = []
errors = []


for x, observation in data_cm.items():
    n = len(observation)
    avg = sum(observation) / n
    variance = sum((xi - avg) ** 2 for xi in observation) / (n - 1)
    std_dev = math.sqrt(variance)

    actual.append(x)
    measured.append(avg)
    errors.append(std_dev)
    
print ("Actual (cm): ", errors)

# Plot with error bars
plt.errorbar(actual, measured, yerr=errors, fmt='o', capsize=5, label="Measured (with std dev)")
plt.plot(actual, actual, 'r--', label="Ideal (y = x)")  # reference line

# Add text labels for each point
for x, y in zip(actual, measured):
    plt.text(x, y + 30, f"{y:.1f}", ha='center', fontsize=9, color="blue")

plt.xlabel("Actual Distance (cm)")
plt.ylabel("Measured Distance (cm)")
plt.title("Measured vs Actual Distance with Standard Deviation")
plt.legend()
plt.grid(True)
plt.show()
