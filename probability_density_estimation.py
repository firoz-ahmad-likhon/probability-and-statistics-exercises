"""Nonparametric Density Estimation:

Nonparametric estimation methods, like Kernel Density Estimation (KDE), do not assume a specific parametric form for
the underlying probability distribution. Instead, they estimate the distribution directly from the data.

Terminology:

1. Density Estimation:
   - The process of estimating the probability distribution of a dataset.

2. Bandwidth (h):
   - A smoothing parameter that controls how wide the kernel function is.
   - Small bandwidth → More details but high variance (overfitting).
   - Large bandwidth → Smoother estimate but might miss important details (underfitting).

3. Kernel Function (K):
   - A function that assigns weights to data points based on their distance from the estimation point.
   - Common choices: Gaussian, Epanechnikov, Uniform, etc.

4. Probability Density Function (PDF):
   - A function that describes the likelihood of a continuous random variable taking a particular value.

5. Empirical Cumulative Distribution Function (ECDF):
   - A function that estimates the probability that a random variable is less than or equal to a given value.

Nonparametric Density Estimation Methods:

1. Histogram Estimation:
   - Divides the data into bins and counts occurrences.
   - When to use:
     - Quick visualization of data distribution.
     - When an approximate density is enough.
   - Disadvantages:
     - Highly dependent on bin width.
     - Not smooth, leading to misleading interpretations.

2. Kernel Density Estimation (KDE):
   - Uses a kernel function to smooth out the data.
   - When to use:
     - When a smooth estimate of the probability distribution is needed.
     - When data is continuous.
   - Disadvantages:
     - Computationally expensive for large datasets.
     - Bandwidth selection is crucial.

3. K-Nearest Neighbors Density Estimation (kNN-DE):
   - Estimates density based on the distance to the k-th nearest neighbor.
   - When to use:
     - When data is sparse or has varying density.
     - When adaptiveness to local density is needed.
   - Disadvantages:
     - Computationally expensive.
     - Sensitive to the choice of k.

4. Empirical Cumulative Distribution Function (ECDF):
   - Estimates the cumulative probability of observed data.
   - When to use:
     - When only cumulative probabilities are needed.
     - When you need a non-smoothed distribution estimate.
   - Disadvantages:
     - Does not give a probability density function (PDF).
     - Discrete jumps instead of smooth curves.

Kernel Density Estimation (KDE) Formula:
    f̂(x) = (1 / (n * h)) * Σ K((x - xi) / h),  for i = 1 to n
Where: n = number of data points
       h = bandwidth (smoothing parameter)
       K(·) = kernel function

Common Kernel Functions:

1. Gaussian Kernel (Most Common)
    K(u) = (1 / sqrt(2π)) * exp(-u² / 2)

2. Epanechnikov Kernel
    K(u) = (3 / 4) * (1 - u²),  for |u| ≤ 1

3. Uniform Kernel
    K(u) = 1/2,  for |u| ≤ 1

Bandwidth Selection Methods:
1. Silverman’s Rule of Thumb:
    h = 1.06 * σ̂ * n^(-1/5) where σ̂ is the sample standard deviation.

2. Scott’s Rule:
    h = n^(-1/5)

3. Cross-validation-based selection:
   - Minimizes estimation error by optimizing bandwidth.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate sample data
data = np.random.normal(0, 1, 1000)

# Compute KDE using Gaussian kernel
kde = gaussian_kde(data, bw_method='scott')

# Generate points for plotting
x_vals = np.linspace(min(data), max(data), 1000)
y_vals = kde(x_vals)

# Plot KDE and histogram
plt.figure(figsize=(8, 5))
sns.histplot(data, bins=30, kde=True, stat="density", alpha=0.3)
plt.plot(x_vals, y_vals, label="KDE Estimate", color="red")
plt.legend()
plt.title("Kernel Density Estimation (KDE)")
plt.show()
