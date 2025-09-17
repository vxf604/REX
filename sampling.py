import random
import math
import matplotlib.pyplot as plt
import numpy as np



def N(x, my, std):
    return 1 / (math.sqrt(2 * math.pi) * std) * math.exp(-0.5 * ((x - my) / std) ** 2)


def pdf(x):
    return 0.3 * N(x, 2.0, 1.0) + 0.4 * N(x, 5.0, 2.0) + 0.3 * N(x, 9.0, 1.0)


def sampling(k):
    return [random.gauss(5,2) for _ in range(k)]


def importance_weighting(samples):   
    weights = []
    normalized_weights = []
    for sample in samples:
        weight = pdf(sample) / N(sample, 5.0, 2.0)
        weights.append(weight)

    for weight in weights:
        normalized_weight = weight / sum(weights)
        normalized_weights.append(normalized_weight)
    return normalized_weights


def resampling(samples, weights, k):
    return random.choices(samples, weights, k=k)


def SIR(k):
    samples = sampling(k)
    weights = importance_weighting(samples)
    resampled = resampling(samples, weights, k)
    return resampled


points = np.linspace(0, 15, 500)
pdf_points = [pdf(x) for x in points]

k20 = SIR(20)
k100 = SIR(100)
k1000 = SIR(1000)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

axes[0].hist(k20, bins=50, range=(0, 15), density=True, label="k=20")
axes[0].plot(points, pdf_points, label="pdf")
axes[0].set_title("k=20")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Probability Density")
axes[0].legend()

axes[1].hist(k100, bins=50, range=(0, 15), density=True, label="k=100")
axes[1].plot(points, pdf_points, label="pdf")
axes[1].set_title("k=100")
axes[1].set_xlabel("x")
axes[1].legend()

axes[2].hist(k1000, bins=50, range=(0, 15), density=True, label="k=1000")
axes[2].plot(points, pdf_points, label="pdf")
axes[2].set_title("k=1000)")
axes[2].set_xlabel("x")
axes[2].legend()

plt.show()
