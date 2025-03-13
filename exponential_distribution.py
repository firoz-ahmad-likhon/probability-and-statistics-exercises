"""Exponential Distribution:
    The exponential distribution is a continuous probability distribution that describes the time between events in a Poisson process.
    It is commonly used in reliability analysis, queuing theory, and survival analysis.

Formula:
    The probability density function (PDF) of the exponential distribution is given by:
    f(x) = λ * e^(-λx),  for x ≥ 0
    where:
        λ = rate parameter (inverse of the mean, 1/μ)
        e = Euler’s number (≈ 2.718)

Statistical Properties:
    - Mean (Expected Value): E[X] = 1/λ
    - Variance: Var(X) = 1/λ²
    - Standard Deviation: σ = 1/λ = E[X] = mean
    - Skewness: 2 (right-skewed distribution)
    - Kurtosis: 9 (leptokurtic, long right tail)

Properties of the Exponential Distribution:
    - Right-skewed (not symmetric)
    - The total area under the curve is 1
    - Memoryless property: P(X > s + t | X > s) = P(X > t)
    - The cumulative distribution function (CDF) is:
        F(x) = 1 - e^(-λx),  for x ≥ 0

Applications:
    - Modeling the time between arrivals of customers at a service center
    - Reliability analysis of electrical components
    - Waiting time in queuing systems
    - Failure rates of mechanical systems

Checks if a problem follows an exponential distribution. Conditions:
    1. Events occur independently.
    2. The rate parameter (λ) remains constant over time.
    3. The probability of an event occurring in a small time interval is proportional to the length of the interval.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import expon

# Define rate parameter λ (inverse of mean)
lambda_ = 0.1

# Generate an exponential distribution
dist = expon(scale=1/lambda_)

print("-----Start calculating probability under points-----")
# Calculate the CDF values at x = 5 and x = 10
cdf_5 = dist.cdf(5)
cdf_10 = dist.cdf(10)

# Calculate the probabilities
prob_between_5_and_10 = cdf_10 - cdf_5
prob_greater_than_10 = 1 - cdf_10  # dist.sf(10)
prob_less_than_5 = cdf_5

print(f"Probability of being between 5 and 10: {prob_between_5_and_10:.2f}")
print(f"Probability of being greater than 10: {prob_greater_than_10:.2f}")
print(f"Probability of being less than 5: {prob_less_than_5:.2f}")
print("-----End calculating probability under points-----\n\n")

print("-----Start Visualizing-----")
# Define the range of values
x_values = np.linspace(0, 50, 1000)

# Calculate the PDF values
pdf_values = dist.pdf(x_values)

# Plot the PDF with highlighted areas
plt.figure(figsize=(8, 6))

sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(5, 10, 1000), dist.pdf(np.linspace(5, 10, 1000)),
                 color='orange', alpha=0.3, label=f'P(5 ≤ X ≤ 10) = {prob_between_5_and_10:.2f}')
plt.fill_between(np.linspace(10, max(x_values), 1000), dist.pdf(np.linspace(10, max(x_values), 1000)),
                 color='green', alpha=0.3, label=f'P(X > 10) = {prob_greater_than_10:.2f}')
plt.fill_between(np.linspace(0, 5, 1000), dist.pdf(np.linspace(0, 5, 1000)),
                 color='red', alpha=0.3, label=f'P(X < 5) = {prob_less_than_5:.2f}')
plt.scatter([5, 10], [dist.pdf(5), dist.pdf(10)], color='black')
plt.text(5, dist.pdf(
    5), f'5\nPDF: {dist.pdf(5):.2f}\n CDF: {cdf_5: .2f}', color='black', ha='right', va='bottom')
plt.text(10, dist.pdf(
    10), f'10\nPDF: {dist.pdf(10):.2f}\n CDF: {cdf_10: .2f}', color='black', ha='left', va='bottom')
plt.xlabel('Time Between Events')
plt.ylabel('PDF')
plt.title('Exponential Distribution Probability')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

plt.show()
print("-----End Visualizing-----")
