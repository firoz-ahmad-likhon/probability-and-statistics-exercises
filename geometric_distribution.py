"""
Geometric Distribution:
    The geometric distribution is a discrete probability distribution that models the number of trials
    required until the first success in independent Bernoulli trials, each with the same probability of success.

Formula:
    The probability mass function (PMF) of the geometric distribution is given by:
    P(X = k) = (1 - p)^(k - 1) * p

where:
    k = number of trials until the first success (k ≥ 1)
    p = probability of success in each trial

Statistical Properties:
    - Mean (Expected Value): E[X] = 1 / p
    - Variance: Var(X) = (1 - p) / p^2
    - Standard Deviation: σ = sqrt((1 - p) / p^2)
    - Mode: 1 (the first trial being a success is the most probable outcome)

Memoryless Property:
    The geometric distribution has the memoryless property:
        P(X > m + n | X > m) = P(X > n)
    This means the probability of success in future trials is independent of past failures.

Applications:
    - Modeling the number of attempts needed to win a game
    - Predicting the number of sales calls before the first successful deal
    - Estimating the number of defective items before finding a functional one
    - Waiting time in queuing theory (discrete time systems)

Checks if a problem follows a geometric distribution. Conditions:
    1. Independent trials.
    2. Each trial has only two outcomes (success/failure).
    3. Constant probability of success (p).
    4. The process continues until the first success.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import geom

# Define geometric distribution parameter
p = 0.3  # Probability of success in each trial

# Create a geometric distribution
dist = geom(p=p)

print("-----Start calculating probability under points-----")
# Define specific points
k1, k2 = 3, 8  # Example values for probability calculations

# Calculate cumulative probabilities
cdf_k1 = dist.cdf(k1)
cdf_k2 = dist.cdf(k2)

# Compute probabilities
prob_between_k1_and_k2 = cdf_k2 - cdf_k1
prob_greater_than_k2 = 1 - cdf_k2  # dist.sf(8)
prob_less_than_k1 = cdf_k1

print(
    f"Probability of getting between {k1} and {k2} trials: {prob_between_k1_and_k2:.2f}")
print(
    f"Probability of getting more than {k2} trials: {prob_greater_than_k2:.2f}")
print(
    f"Probability of getting fewer than {k1} trials: {prob_less_than_k1:.2f}")
print("-----End calculating probability under points-----\n\n")

print("-----Start Visualizing-----")
# Define range for geometric values
x_values = np.arange(1, 20)

# Compute PMF values
pmf_values = dist.pmf(x_values)

plt.figure(figsize=(8, 6))

# Plot PMF with highlighted probabilities for k1 and k2
sns.barplot(x=x_values, y=pmf_values, color='blue', alpha=0.6)
plt.bar(np.arange(k1, k2 + 1), dist.pmf(np.arange(k1, k2 + 1)), color='orange', alpha=0.6,
        label=f'P({k1} ≤ X ≤ {k2}) = {prob_between_k1_and_k2:.2f}')
plt.bar(np.arange(k2 + 1, 20), dist.pmf(np.arange(k2 + 1, 20)), color='green', alpha=0.6,
        label=f'P(X > {k2}) = {prob_greater_than_k2:.2f}')
plt.bar(np.arange(1, k1), dist.pmf(np.arange(1, k1)), color='red', alpha=0.6,
        label=f'P(X < {k1}) = {prob_less_than_k1:.2f}')
plt.xlabel('Number of Trials Until First Success (X)')
plt.ylabel('PMF')
plt.title('Geometric Distribution')
plt.legend()
plt.grid(axis='y')

plt.show()
print("-----End Visualizing-----")
