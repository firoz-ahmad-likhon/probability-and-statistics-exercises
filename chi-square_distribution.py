"""
Chi-Square Distribution:
    The Chi-Square distribution is a probability distribution that is used primarily in hypothesis testing, particularly for tests of independence and goodness of fit. It is a special case of the gamma distribution and is characterized by its positive skew and non-negative values.

Statistical Properties:
    - Mean: E(χ²) = df
    - Median (Approximate): df - (2/3)
    - Variance: Var(χ²) = 2df
    - Standard Deviation: std(χ²) = √(2df)

When to Use the Chi-Square Distribution:
    - When testing the goodness of fit of observed data to an expected distribution.
    - When assessing the independence of two categorical variables in a contingency table.
    - When comparing the variances of different samples.

Convergence with Normal Distribution:
    - As the degrees of freedom (df) increase, the Chi-Square distribution becomes more symmetric and approaches the normal distribution.
    - For df > 30, the Chi-Square distribution is approximately normal.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

data = np.array([1, 3, 3, 6, 6, 8, 9, 2, 5, 6, 5, 3, 2])
df = len(data) - 1  # Degree of freedom

dist = chi2(df)  # Chi-Square distribution

print("-----Start calculating probability under points-----")
# Calculate the CDF values at 2 and 5
cdf_2 = dist.cdf(2)
cdf_5 = dist.cdf(5)

# Calculate the probabilities
prob_between_2_and_5 = cdf_5 - cdf_2
prob_greater_than_5 = 1 - cdf_5  # dist.sf(5))
prob_less_than_2 = cdf_2

print(f"Probability of being between 2 and 5: {prob_between_2_and_5:.2f}")
print(f"Probability of being greater than 5: {prob_greater_than_5:.2f}")
print(f"Probability of being less than 2: {prob_less_than_2:.2f}")
print("-----End calculating probability under points-----\n\n")

print("-----Start calculating probability under percentiles-----")
# Find the values at the 15th and 65th percentiles
point_of_15 = dist.ppf(0.15)
point_of_65 = dist.ppf(0.65)

# Calculate the CDF values at these points
cdf_point_of_15 = dist.cdf(point_of_15)
cdf_point_of_65 = dist.cdf(point_of_65)

# Calculate the probabilities
prob_between_point_of_15_and_point_of_65 = cdf_point_of_65 - cdf_point_of_15
prob_greater_than_point_of_65 = 1 - cdf_point_of_65  # dist.sf(dist.ppf(.65))
prob_less_than_point_of_15 = cdf_point_of_15

print(
    f"Probability of being between 15th and 65th percentile: {prob_between_point_of_15_and_point_of_65:.2f}")
print(
    f"Probability of being greater than 65th percentile: {prob_greater_than_point_of_65:.2f}")
print(
    f"Probability of being less than 15th percentile: {prob_less_than_point_of_15:.2f}")
print("-----End calculating probability under percentiles-----\n\n")

print("-----Start Visualizing-----")
# Define the range of values
x_values = np.linspace(0, 15, 1000)

# Calculate the PDF values
pdf_values = dist.pdf(x_values)

# Plot the PDF and highlight areas
plt.figure(figsize=(12, 12))

# Plot 1: PDF, CDF with highlighted areas for 2 and 5
plt.subplot(1, 2, 1)
sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(2, 5, 1000), dist.pdf(np.linspace(2, 5, 1000)),
                 color='orange', alpha=0.3, label=f'P(2 ≤ X ≤ 5) = {prob_between_2_and_5:.2f}')
plt.fill_between(np.linspace(5, max(x_values), 1000), dist.pdf(np.linspace(5, max(x_values), 1000)),
                 color='green', alpha=0.3, label=f'P(X > 5) = {prob_greater_than_5:.2f}')
plt.fill_between(np.linspace(min(x_values), 2, 1000), dist.pdf(np.linspace(min(x_values), 2, 1000)),
                 color='red', alpha=0.3, label=f'P(X < 2) = {prob_less_than_2:.2f}')
plt.scatter([2, 5], [dist.pdf(2), dist.pdf(5)], color='black')
plt.text(2, dist.pdf(
    2), f'2\nPDF: {dist.pdf(2):.2f}\n CDF: {cdf_2: .2f}', color='black', ha='right', va='bottom')
plt.text(5, dist.pdf(
    5), f'5\nPDF: {dist.pdf(5):.2f}\n CDF: {cdf_5: .2f}', color='black', ha='left', va='bottom')
plt.text(df - 1, 0.04, f'{prob_between_2_and_5:.2f}',
         color='orange', ha='center')
plt.text(7, 0.01, f'{prob_greater_than_5:.2f}', color='green', ha='center')
plt.text(1.5, 0.01, f'{prob_less_than_2:.2f}', color='red', ha='center')
plt.xlabel('Value')
plt.ylabel('PDF')
plt.title('Probability Under Points (Chi-Square)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

# Plot 2: PDF, PPF with highlighted areas for 15% and 65%
plt.subplot(1, 2, 2)
sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(point_of_15, point_of_65, 1000), dist.pdf(np.linspace(point_of_15, point_of_65, 1000)),
                 color='orange', alpha=0.3, label=f'P(15th ≤ X ≤ 65th) = {prob_between_point_of_15_and_point_of_65:.2f}')
plt.fill_between(np.linspace(point_of_65, max(x_values), 1000), dist.pdf(np.linspace(point_of_65, max(
    x_values), 1000)), color='green', alpha=0.3, label=f'P(X > 65th) = {prob_greater_than_point_of_65:.2f}')
plt.fill_between(np.linspace(min(x_values), point_of_15, 1000), dist.pdf(np.linspace(min(
    x_values), point_of_15, 1000)), color='red', alpha=0.3, label=f'P(X < 15th) = {prob_less_than_point_of_15:.2f}')
plt.scatter([point_of_15, point_of_65], [dist.pdf(
    point_of_15), dist.pdf(point_of_65)], color='black')
plt.text(point_of_15, dist.pdf(point_of_15),
         f'15th\nPDF: {dist.pdf(point_of_15):.2f}\n CDF: {cdf_point_of_15: .2f}', color='black', ha='right', va='bottom')
plt.text(point_of_65, dist.pdf(point_of_65),
         f'65th\nPDF: {dist.pdf(point_of_65):.2f}\n CDF: {cdf_point_of_65: .2f}', color='black', ha='left', va='bottom')
plt.text(df - 1, 0.04,
         f'{prob_between_point_of_15_and_point_of_65:.2f}', color='orange', ha='center')
plt.text(np.max(x_values) - 1, 0.01,
         f'{prob_greater_than_point_of_65:.2f}', color='green', ha='center')
plt.text(np.min(x_values) + 1, 0.01,
         f'{prob_less_than_point_of_15:.2f}', color='red', ha='center')
plt.xlabel('Value')
plt.ylabel('PDF')
plt.title('Probability Under Percentiles (Chi-Square)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

plt.show()
print("-----End Visualizing-----")
