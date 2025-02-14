"""
t-Distribution:
    The t-distribution (Student's t-distribution) is a probability distribution that is similar in shape to the normal distribution but has heavier tails, meaning it accounts for more variability in small samples.

When to Use the t-Distribution:
    - When working with small sample sizes (n < 30).
    - When the population standard deviation (σ) is unknown.

Convergence with Normal Distribution:
    - As the degrees of freedom (df = n - 1) increase, the t-distribution approaches the standard normal distribution (Z-distribution).
    - For df > 30, the difference between the t-distribution and normal distribution becomes negligible.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t

speed = np.array([94, 85, 87, 88, 111, 85, 103, 87, 94, 78, 77, 85, 85])
n = len(speed)
df = n - 1  # Degree of freedom
mean = np.mean(speed)
std = np.std(speed, ddof=1)
se = std / np.sqrt(n)  # Standard error from CLT

dist = t(df, loc=mean, scale=se)  # Student's t-distribution

print("-----Start calculating probability under points-----")
# Calculate the CDF values at 85 and 94
cdf_85 = dist.cdf(85)
cdf_94 = dist.cdf(94)

# Calculate the probabilities
prob_between_85_and_94 = cdf_94 - cdf_85
prob_greater_than_94 = 1 - cdf_94  # dist.sf(94)
prob_less_than_85 = cdf_85

print(f"Probability of being between 85 and 94: {prob_between_85_and_94:.2f}")
print(f"Probability of being greater than 94: {prob_greater_than_94:.2f}")
print(f"Probability of being less than 85: {prob_less_than_85:.2f}")
print("-----End calculating probability under points-----\n\n")

print("-----Start calculating probability under percentiles-----")
# Find the values at the 34th and 84th percentiles
point_of_34 = dist.ppf(0.34)
point_of_84 = dist.ppf(0.84)

# Calculate the CDF values at these points
cdf_point_of_34 = dist.cdf(point_of_34)
cdf_point_of_84 = dist.cdf(point_of_84)

# Calculate the probabilities
prob_between_point_of_34_and_point_of_84 = cdf_point_of_84 - cdf_point_of_34
prob_greater_than_point_of_84 = 1 - cdf_point_of_84  # dist.sf(dist.ppf(.84))
prob_less_than_point_of_34 = cdf_point_of_34

print(
    f"Probability of being between 34th and 84th percentile: {prob_between_point_of_34_and_point_of_84:.2f}")
print(
    f"Probability of being greater than 84th percentile: {prob_greater_than_point_of_84:.2f}")
print(
    f"Probability of being less than 34th percentile: {prob_less_than_point_of_34:.2f}")
print("-----End calculating probability under percentiles-----\n\n")

print("-----Start Visualizing-----")
# Define the range of values
x_values = np.linspace(min(speed) - 10, max(speed) + 20, 1000)

# Calculate the PDF values
pdf_values = dist.pdf(x_values)

# Plot the PDF and highlight areas
plt.figure(figsize=(12, 12))

# Plot 1: PDF, CDF with highlighted areas for 85 and 94
plt.subplot(1, 2, 1)
sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(85, 94, 1000), dist.pdf(np.linspace(85, 94, 1000)),
                 color='orange', alpha=0.3, label=f'P(85 ≤ X ≤ 94) = {prob_between_85_and_94:.2f}')
plt.fill_between(np.linspace(94, max(speed)+10, 1000), dist.pdf(np.linspace(94, max(speed)+10, 1000)),
                 color='green', alpha=0.3, label=f'P(X > 94) = {prob_greater_than_94:.2f}')
plt.fill_between(np.linspace(min(speed)-10, 85, 1000), dist.pdf(np.linspace(min(speed)-10,
                 85, 1000)), color='red', alpha=0.3, label=f'P(X < 85) = {prob_less_than_85:.2f}')
plt.scatter([85, 94], [dist.pdf(85), dist.pdf(94)], color='black')
plt.text(85, dist.pdf(
    85), f'85\nPDF: {dist.pdf(85):.2f}\n CDF: {cdf_85: .2f}', color='black', ha='right', va='bottom')
plt.text(94, dist.pdf(
    94), f'94\nPDF: {dist.pdf(94):.2f}\n CDF: {cdf_94: .2f}', color='black', ha='left', va='bottom')
plt.text(92.5, 0.02, f'{prob_between_85_and_94:.2f}',
         color='orange', ha='center')
plt.text(105, 0.005, f'{prob_greater_than_94:.2f}', color='green', ha='center')
plt.text(75, 0.005, f'{prob_less_than_85:.2f}', color='red', ha='center')
plt.xlabel('Speed')
plt.ylabel('PDF')
plt.title('Probability Under Points (t-distribution)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

# Plot 2: PDF, PPF with highlighted areas for 34% and 84%
plt.subplot(1, 2, 2)
sns.lineplot(x=x_values, y=pdf_values, color='blue', label='PDF')
plt.fill_between(np.linspace(point_of_34, point_of_84, 1000), dist.pdf(np.linspace(point_of_34, point_of_84, 1000)),
                 color='orange', alpha=0.3, label=f'P(34th ≤ X ≤ 84th) = {prob_between_point_of_34_and_point_of_84:.2f}')
plt.fill_between(np.linspace(point_of_84, max(x_values), 1000), dist.pdf(np.linspace(point_of_84, max(
    x_values), 1000)), color='green', alpha=0.3, label=f'P(X > 84th) = {prob_greater_than_point_of_84:.2f}')
plt.fill_between(np.linspace(min(x_values), point_of_34, 1000), dist.pdf(np.linspace(min(
    x_values), point_of_34, 1000)), color='red', alpha=0.3, label=f'P(X < 34th) = {prob_less_than_point_of_34:.2f}')
plt.scatter([point_of_34, point_of_84], [dist.pdf(
    point_of_34), dist.pdf(point_of_84)], color='black')
plt.text(point_of_34, dist.pdf(point_of_34),
         f'34th\nPDF: {dist.pdf(point_of_34):.2f}\n CDF: {cdf_point_of_34: .2f}', color='black', ha='right', va='bottom')
plt.text(point_of_84, dist.pdf(point_of_84),
         f'84th\nPDF: {dist.pdf(point_of_84):.2f}\n CDF: {cdf_point_of_84: .2f}', color='black', ha='left', va='bottom')
plt.text(np.mean(x_values) - 7, 0.02,
         f'{prob_between_point_of_34_and_point_of_84:.2f}', color='orange', ha='center')
plt.text(np.max(x_values) - 26, 0.005,
         f'{prob_greater_than_point_of_84:.2f}', color='green', ha='center')
plt.text(np.min(x_values) + 10, 0.005,
         f'{prob_less_than_point_of_34:.2f}', color='red', ha='center')
plt.xlabel('Speed')
plt.ylabel('PDF')
plt.title('Probability Under Percentiles (t-distribution)')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)
plt.tight_layout()

plt.show()
print("-----End Visualizing-----")
