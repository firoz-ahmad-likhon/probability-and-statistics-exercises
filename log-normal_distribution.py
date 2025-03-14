"""Log-Normal Distribution:
    The log-normal distribution is a continuous probability distribution of a random variable whose logarithm is
    normally distributed. It models right-skewed data where values are positive and multiplicative rather than additive.

Formula:
    The probability density function (PDF) of the log-normal distribution is given by:
    f(x) = (1 / (xσ sqrt(2π))) * e^(- (ln(x) - μ)² / (2σ²))
where:
    μ = mean of the natural logarithm of x
    σ² = variance of the natural logarithm of x
    σ = standard deviation of the natural logarithm of x
    e = Euler’s number (≈ 2.718)
    π = Pi (≈ 3.1416)

Statistical Properties:
    - Mean: E[X] = e^(μ + σ²/2)
    - Variance: Var(X) = (e^(σ²) - 1) * e^(2μ + σ²)
    - Mode: e^(μ - σ²)
    - Skewness: (e^(σ²) + 2) * sqrt(e^(σ²) - 1)
    - Kurtosis: e^(4σ²) + 2e^(3σ²) + 3e^(2σ²) - 6

Log-Normal Transformation:
    If X follows a log-normal distribution, then:
        Y = ln(X) ~ N(μ, σ²)
    follows a normal distribution with mean μ and variance σ².

Properties of the Log-Normal Distribution:
    - Always positive (X > 0)
    - Right-skewed distribution
    - Mean > Median > Mode
    - Skewness and kurtosis depend on σ²

Applications:
    - Modeling stock prices and financial returns
    - Analyzing income distribution and economic data
    - Modeling size of biological organisms (e.g., bacteria growth)
    - Reliability analysis and failure times
    - Environmental data modeling (e.g., pollutant concentration)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import lognorm

# Sample data representing a log-normal distribution
shape = 0.954  # Standard deviation of log(X)
scale = np.exp(4.5)  # Mean of log(X) transformed back

# Create a log-normal distribution
lognorm_dist = lognorm(s=shape, scale=scale)

print("-----Start calculating probability under points-----")
# Calculate the CDF values at 100 and 200
cdf_100 = lognorm_dist.cdf(100)
cdf_200 = lognorm_dist.cdf(200)

# Calculate the probabilities
prob_between_100_and_200 = cdf_200 - cdf_100
prob_greater_than_200 = 1 - cdf_200
prob_less_than_100 = cdf_100

print(
    f"Probability of being between 100 and 200: {prob_between_100_and_200:.2f}")
print(f"Probability of being greater than 200: {prob_greater_than_200:.2f}")
print(f"Probability of being less than 100: {prob_less_than_100:.2f}")
print("-----End calculating probability under points-----\n\n")

print("-----Start Visualizing-----")
# Define the range of values
x_values = np.linspace(10, 500, 1000)

# Calculate the PDF values
pdf_values = lognorm_dist.pdf(x_values)

# Plot the PDF and highlight areas
plt.figure(figsize=(10, 6))
sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(100, 200, 1000), lognorm_dist.pdf(np.linspace(100, 200, 1000)),
                 color='orange', alpha=0.3, label=f'P(100 ≤ X ≤ 200) = {prob_between_100_and_200:.2f}')
plt.fill_between(np.linspace(200, max(x_values), 1000), lognorm_dist.pdf(np.linspace(200, max(x_values), 1000)),
                 color='green', alpha=0.3, label=f'P(X > 200) = {prob_greater_than_200:.2f}')
plt.fill_between(np.linspace(min(x_values), 100, 1000), lognorm_dist.pdf(np.linspace(min(x_values), 100, 1000)),
                 color='red', alpha=0.3, label=f'P(X < 100) = {prob_less_than_100:.2f}')
plt.scatter([100, 200], [lognorm_dist.pdf(100),
            lognorm_dist.pdf(200)], color='black')
plt.xlabel('X')
plt.ylabel('PDF')
plt.title('Log-Normal Distribution')
plt.legend()
plt.grid(True)
plt.show()
print("-----End Visualizing-----")
