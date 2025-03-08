"""
Poisson Distribution:
    The Poisson distribution is a discrete probability distribution that models the number of occurrences
    of an event in a fixed interval of time or space, given a known constant mean rate of occurrence.

Formula:
    The probability mass function (PMF) of the Poisson distribution is given by:
    P(X = k) = (λ^k * e^(-λ)) / k!
where:
    λ = average number of occurrences in the interval
    k = number of occurrences (0 ≤ k)
    e = Euler’s number (≈2.718)

Statistical Properties:
    - Mean (Expected Value): E[X] = λ
    - Variance: Var(X) = λ
    - Standard Deviation: σ = sqrt(λ)
    - Mode: ⌊λ⌋ (or ⌊λ⌋ and ⌊λ⌋ - 1 if λ is an integer)

Convergence to Normal Distribution:
    By the Central Limit Theorem (CLT), as λ increases, the Poisson distribution approximates a normal distribution
    N(μ, σ²) with:
        μ = λ
        σ² = λ
    For a good normal approximation, λ ≥ 30 is a common rule of thumb.

Applications:
    - Modeling the number of calls received at a call center per hour
    - Predicting the number of emails arriving in an inbox per day
    - Analyzing traffic accidents in a city per week
    - Biological modeling (e.g., mutations in DNA over time)
    - Estimating the number of defects in a production process

Checks if a problem follows a Poisson distribution. Conditions:
    1. Events occur independently.
    2. Events occur at a constant average rate.
    3. Two events cannot occur at the exact same time.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# Define Poisson distribution parameter
λ = 10  # Average rate of occurrences

# Create a Poisson distribution
dist = poisson(λ)

print("-----Start calculating probability under points-----")
# Define specific points
k1, k2 = 5, 15  # Example values for probability calculations

# Calculate cumulative probabilities
cdf_k1 = dist.cdf(k1)
cdf_k2 = dist.cdf(k2)

# Compute probabilities
prob_between_k1_and_k2 = cdf_k2 - cdf_k1
prob_greater_than_k2 = 1 - cdf_k2  # dist.sf(15)
prob_less_than_k1 = cdf_k1

print(
    f"Probability of getting between {k1} and {k2} occurrences: {prob_between_k1_and_k2:.2f}")
print(
    f"Probability of getting more than {k2} occurrences: {prob_greater_than_k2:.2f}")
print(
    f"Probability of getting fewer than {k1} occurrences: {prob_less_than_k1:.2f}")
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
# Define range for Poisson values
x_values = np.arange(0, λ * 2 + 1)

# Compute PMF values
pmf_values = dist.pmf(x_values)

plt.figure(figsize=(12, 12))

# Plot 1: PMF with highlighted probabilities for k1 and k2
plt.subplot(1, 2, 1)
sns.barplot(x=x_values, y=pmf_values, color='blue', alpha=0.6)
plt.bar(np.arange(k1, k2 + 1), dist.pmf(np.arange(k1, k2 + 1)), color='orange', alpha=0.6,
        label=f'P({k1} ≤ X ≤ {k2}) = {prob_between_k1_and_k2:.2f}')
plt.bar(np.arange(k2 + 1, λ * 2 + 1), dist.pmf(np.arange(k2 + 1, λ * 2 + 1)), color='green', alpha=0.6,
        label=f'P(X > {k2}) = {prob_greater_than_k2:.2f}')
plt.bar(np.arange(0, k1), dist.pmf(np.arange(0, k1)), color='red', alpha=0.6,
        label=f'P(X < {k1}) = {prob_less_than_k1:.2f}')
plt.xlabel('Number of Occurrences (X)')
plt.ylabel('PMF')
plt.title('Probability Under Points (Poisson Distribution)')
plt.legend()
plt.grid(axis='y')

# Plot 2: PMF with highlighted probabilities for percentiles
plt.subplot(1, 2, 2)
sns.barplot(x=x_values, y=pmf_values, color='blue', alpha=0.6)
plt.bar(np.arange(point_of_25, point_of_75 + 1), dist.pmf(np.arange(point_of_25, point_of_75 + 1)), color='orange', alpha=0.6,
        label=f'P(25th ≤ X ≤ 75th) = {prob_between_point_of_25_and_point_of_75:.2f}')
plt.bar(np.arange(point_of_75 + 1, λ * 2 + 1), dist.pmf(np.arange(point_of_75 + 1, λ * 2 + 1)), color='green', alpha=0.6,
        label=f'P(X > 75th) = {prob_greater_than_point_of_75:.2f}')
plt.bar(np.arange(0, point_of_25), dist.pmf(np.arange(0, point_of_25)), color='red', alpha=0.6,
        label=f'P(X < 25th) = {prob_less_than_point_of_25:.2f}')
plt.xlabel('Number of Occurrences (X)')
plt.ylabel('PMF')
plt.title('Probability Under Percentiles (Poisson Distribution)')
plt.legend()
plt.grid(axis='y')

plt.tight_layout()
plt.show()
print("-----End Visualizing-----")
