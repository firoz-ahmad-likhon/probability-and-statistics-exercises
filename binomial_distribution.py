"""
Binomial Distribution:
    The binomial distribution is a discrete probability distribution that models the number of successes
    in a fixed number of independent Bernoulli trials, each with the same probability of success.
Formula:
    The probability mass function (PMF) of the binomial distribution is given by:
    P(X = k) = C(n, k) * p^k * (1 - p)^(n - k)
where:
    n = number of trials
    k = number of successes (0 ≤ k ≤ n)
    p = probability of success in each trial
    C(n, k) = n! / (k!(n - k)!) (binomial coefficient)  # Combinations formula

Statistical Properties:
    - Mean (Expected Value): E[X] = n * p
    - Variance: Var(X) = n * p * (1 - p)
    - Standard Deviation: σ = sqrt(n * p * (1 - p))
    - Mode: ⌊(n + 1) * p⌋

Convergence to Normal Distribution:
    By the Central Limit Theorem (CLT), as the number of trials (n) increases, the binomial distribution
    approximates a normal distribution N(μ, σ²) with:
        μ = n * p
        σ² = n * p * (1 - p)
    For a good normal approximation, np ≥ 10 and n(1 - p) ≥ 10 should hold.

Applications:
    - Modeling the number of defective items in a batch
    - Predicting the probability of passing an exam given a certain number of attempts
    - Analyzing the results of surveys and experiments
    - Genetic probability calculations in biology
    - Quality control in manufacturing

Checks if a problem follows a binomial distribution. Conditions:
    1. Fixed number of trials (n).
    2. Each trial has only two outcomes (success/failure).
    3. Constant probability of success (p).
    4. Trials are independent (default: True).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom

# Define binomial distribution parameters
n = 20   # Number of trials
p = 0.4  # Probability of success in each trial

# Create a binomial distribution
dist = binom(n=n, p=p)

print("-----Start calculating probability under points-----")
# Define specific points
k1, k2 = 5, 12  # Example values for probability calculations

# Calculate cumulative probabilities
cdf_k1 = dist.cdf(k1)
cdf_k2 = dist.cdf(k2)

# Compute probabilities
prob_between_k1_and_k2 = cdf_k2 - cdf_k1
prob_greater_than_k2 = 1 - cdf_k2  # dist.sf(12)
prob_less_than_k1 = cdf_k1

print(
    f"Probability of getting between {k1} and {k2} successes: {prob_between_k1_and_k2:.2f}")
print(
    f"Probability of getting more than {k2} successes: {prob_greater_than_k2:.2f}")
print(
    f"Probability of getting fewer than {k1} successes: {prob_less_than_k1:.2f}")
print("-----End calculating probability under points-----\n\n")

print("-----Start calculating probability under percentiles-----")
# Find the values at the 25th and 75th percentiles
point_of_25 = dist.ppf(0.25)
point_of_75 = dist.ppf(0.75)

# Calculate CDF values at these percentiles
cdf_point_of_25 = dist.cdf(point_of_25)
cdf_point_of_75 = dist.cdf(point_of_75)

# Compute probabilities
prob_between_point_of_25_and_point_of_75 = cdf_point_of_75 - cdf_point_of_25
prob_greater_than_point_of_75 = 1 - cdf_point_of_75  # dist.sf(dist.ppf(.75))
prob_less_than_point_of_25 = cdf_point_of_25

print(
    f"Probability of getting between 25th and 75th percentile: {prob_between_point_of_25_and_point_of_75:.2f}")
print(
    f"Probability of getting more than 75th percentile: {prob_greater_than_point_of_75:.2f}")
print(
    f"Probability of getting fewer than 25th percentile: {prob_less_than_point_of_25:.2f}")
print("-----End calculating probability under percentiles-----\n\n")

print("-----Start Visualizing-----")
# Define range for binomial values
x_values = np.arange(0, n + 1)

# Compute PMF values
pmf_values = dist.pmf(x_values)

plt.figure(figsize=(12, 12))

# Plot 1: PMF with highlighted probabilities for k1 and k2
plt.subplot(1, 2, 1)
sns.barplot(x=x_values, y=pmf_values, color='blue', alpha=0.6)
plt.bar(np.arange(k1, k2 + 1), dist.pmf(np.arange(k1, k2 + 1)), color='orange', alpha=0.6,
        label=f'P({k1} ≤ X ≤ {k2}) = {prob_between_k1_and_k2:.2f}')
plt.bar(np.arange(k2 + 1, n + 1), dist.pmf(np.arange(k2 + 1, n + 1)), color='green', alpha=0.6,
        label=f'P(X > {k2}) = {prob_greater_than_k2:.2f}')
plt.bar(np.arange(0, k1), dist.pmf(np.arange(0, k1)), color='red', alpha=0.6,
        label=f'P(X < {k1}) = {prob_less_than_k1:.2f}')
plt.xlabel('Number of Successes (X)')
plt.ylabel('PMF')
plt.title('Probability Under Points (Binomial Distribution)')
plt.legend()
plt.grid(axis='y')

# Plot 2: PMF with highlighted probabilities for percentiles
plt.subplot(1, 2, 2)
sns.barplot(x=x_values, y=pmf_values, color='blue', alpha=0.6)
plt.bar(np.arange(point_of_25, point_of_75 + 1), dist.pmf(np.arange(point_of_25, point_of_75 + 1)), color='orange', alpha=0.6,
        label=f'P(25th ≤ X ≤ 75th) = {prob_between_point_of_25_and_point_of_75:.2f}')
plt.bar(np.arange(point_of_75 + 1, n + 1), dist.pmf(np.arange(point_of_75 + 1, n + 1)), color='green', alpha=0.6,
        label=f'P(X > 75th) = {prob_greater_than_point_of_75:.2f}')
plt.bar(np.arange(0, point_of_25), dist.pmf(np.arange(0, point_of_25)), color='red', alpha=0.6,
        label=f'P(X < 25th) = {prob_less_than_point_of_25:.2f}')
plt.xlabel('Number of Successes (X)')
plt.ylabel('PMF')
plt.title('Probability Under Percentiles (Binomial Distribution)')
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()
print("-----End Visualizing-----")
