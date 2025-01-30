"""
Central Limit Theorem (CLT):

1. Definition:
   - The CLT states that, for a sufficiently large sample size, the sampling distribution of the sample mean or proportion will approximate a normal distribution, regardless of the original population distribution.

2. Continuous Data:
   - If the population standard deviation (σ) is known, and the sample size (n) is large (n ≥ 30):
     - Mean of Sampling Distribution (μx̄): Equals the population mean (μ).
     - Standard Error (SE): σₓ̄ = σ / √n.
     - Sampling Distribution Shape: Normal distribution.
   - If the population standard deviation (σ) is unknown, or the sample size (n) is lower (n < 30):
     - Mean of Sampling Distribution (μx̄): Equals the sample mean (x̄).
     - Standard Error (SE): σₓ̄ = s / √n.
     - Sampling Distribution Shape: t-distribution (not normal) with n−1 degrees of freedom

3. Categorical Data (Proportions):
   - If the sample size is sufficiently large to satisfy:
     - n * p ≥ 10, and
     - n * (1 - p) ≥ 10, where p is the population proportion:
     - Mean of Sampling Distribution (p̂): Equals the population proportion (p).
     - Standard Error (SE): SE = √(p * (1 - p) / n).
     - Sampling Distribution Shape: Normal distribution.

4. Why CLT Matters:
   - It allows us to make inferences about the population using sample data.
   - Enables hypothesis testing, construction of confidence intervals, and other statistical analyses.

5. Key Points to Note:
   - The larger the sample size, the closer the sampling distribution is to a normal distribution.
   - For small sample sizes (n < 30), the population must follow a normal distribution for the sample mean to be normally distributed.
   - CLT applies to the mean and proportion, not to individual data points.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
population_mean = 50
sample_size = 40
num_samples = 1000

# Generate population data (exponential distribution for skewed data)
population_data = np.random.exponential(scale=population_mean, size=10000)

print("-----Start statistics for population from data-----")
population_mean_actual = np.mean(population_data)
population_std_actual = np.std(population_data)

print(f'Population Mean: {population_mean_actual:.2f}')
print(f'Population Std Dev: {population_std_actual:.2f}')
print("-----End statistics for population from data-----\n\n")

print("-----Start statistics for sample mean-----")
# Collect sample means
sample_means = []
for _ in range(num_samples):
    sample = np.random.choice(population_data, sample_size)
    sample_means.append(np.mean(sample))

# Calculate statistics for sample means
mean_of_sample_means = np.mean(sample_means)
std_of_sample_means = np.std(sample_means)

print(f'Mean of Sample Means: {mean_of_sample_means:.2f}')
print(f'Std Dev of Sample Means: {std_of_sample_means:.2f}')
print("-----End statistics for sample mean-----\n\n")

print("-----Start statistics by CLT-----")
# Calculate statistics
clt_mean = population_mean_actual
clt_std = population_std_actual / np.sqrt(sample_size)

print(f'Mean by CLT: {clt_mean:.2f}')
print(f'Std Dev by CLT: {clt_std:.2f}')
print("-----End statistics by CLT-----\n\n")

print("-----Start Visualizing-----")
# Plot the population distribution
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.hist(population_data, bins=30, density=True,
         alpha=0.6, color='skyblue', edgecolor='black')
x_pop = np.linspace(min(population_data), max(population_data), 100)
pdf_pop = norm.pdf(x_pop, population_mean_actual, population_std_actual)
plt.plot(x_pop, pdf_pop, 'k', linewidth=2, label='PDF')
plt.title('Skewed Population Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend(
    [f'Mean: {population_mean_actual:.2f}\nStd Dev: {population_std_actual:.2f}'])

# Plot the sampling distribution of the sample mean
plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=30, density=True, alpha=0.6,
         color='lightgreen', edgecolor='black')
plt.title('Sampling Distribution')
plt.xlabel('Sample Mean')
plt.ylabel('Density')

# Overlay the normal distribution
x_sample = np.linspace(min(sample_means), max(sample_means), 100)
pdf_sample = norm.pdf(x_sample, mean_of_sample_means, std_of_sample_means)
plt.plot(x_sample, pdf_sample, 'k', linewidth=2, label='Normal PDF')
plt.legend(
    [f'Mean: {mean_of_sample_means:.2f}\nStd Dev: {std_of_sample_means:.2f}'])

plt.tight_layout()
plt.show()
print("-----End Visualizing-----")
