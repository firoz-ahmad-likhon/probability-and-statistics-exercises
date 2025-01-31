"""
A hypothesis error occurs when the decision made in a hypothesis test is incorrect. There are two types of hypothesis errors:
    1. Type I error (α): Rejecting the null hypothesis when it is actually true.
    2. Type II error (β): Failing to reject the null hypothesis when it is actually false.

The power of a test is the probability that the test will correctly reject a false null hypothesis (H₀). In other words, it is the probability of not committing a Type II error (β).

Notation:
    P(Reject H₀ | H₀ is true) = α
    P(Fail to reject H₀ | H₁ is true) = β
    P(Reject H₀ | H₁ is true) = 1 - β

Example:
    P(Reject the claim that the battery lasts 500 hours ∣ the battery does last 500 hours) = α
    P(Fail to reject the claim that the battery lasts 500 hours ∣ the battery does not last 500 hours) = β
    P(Reject the claim that the battery lasts 500 hours ∣ the battery does not last 500 hours) = 1 - β

In practice, the power to be at least 0.80, meaning there’s an 80% chance of correctly rejecting a false null hypothesis. If the power is too low, then:
1. Increase the sample size.
2. Increase the significance level (though this increases the risk of a Type I error).
3. Reduce variability (if possible).
"""
from scipy import stats
import math
import numpy as np


def validate_estimation_inputs(p: float, n: int) -> tuple[bool, str]:
    """Check the condition if it meets the constraints.

    :param p: Proportion
    :param n: Sample size
    :return: Tuple of boolean and string message
    """
    success = n * p
    fail = n * (1 - p)
    if success < 10 or fail < 10:
        return False, "For proportion estimation, success and fail must both be at least 10."

    return True, "Inputs are valid."


def z_distribution(mean: float, se: float, alpha: float, tail: str) -> tuple[np.float64, np.float64]:
    """Estimate using Z Distribution.

    :param mean: Sample Mean (float)
    :param se: Standard Error (float)
    :param alpha: Significance level (float), e.g., 0.05 for 5%
    :param tail: Tail of the distribution (str), either "two" or "left" or "right"
    :return: Tuple containing:
        - Confidence Interval (tuple of two floats)
    """
    if tail == "two":
        z_critical = stats.norm.ppf(1 - alpha / 2)
    elif tail == "left":
        z_critical = abs(stats.norm.ppf(alpha))
    else:
        z_critical = stats.norm.ppf(1 - alpha)

    moe = z_critical * se  # Margin of Error
    return mean - moe, mean + moe  # Confidence Interval


def t_distribution(mean: float, se: float, alpha: float, n: int, tail: str) -> tuple[np.float64, np.float64]:
    """Estimate using T Distribution.

    :param mean: Sample mean (float)
    :param se: Standard Error (float)
    :param alpha: Significance level (float), e.g., 0.05 for 5%
    :param n: Sample size (int)
    :param tail: Tail of the distribution (str), either "two" or "left" or "right"
    :return: Tuple containing:
        - Confidence Interval (tuple of two floats)
    """
    df = n - 1  # Degree of Freedom

    if tail == "two":
        t_critical = stats.t.ppf(1 - alpha / 2, df)
    elif tail == "left":
        t_critical = abs(stats.t.ppf(alpha, df))
    else:
        t_critical = stats.t.ppf(1 - alpha, df)

    moe = t_critical * se  # Margin of Error
    return mean - moe, mean + moe  # Confidence Interval


def calculate_mean_estimation(mean_null: float, mean_alt: float, n: int, std: float, is_population_std: bool, tail: str, alpha: float = 0.05, stat_type: str = "mean") -> tuple[float, np.float64, np.float64]:
    """Estimate population parameters using the mean and standard deviation.

    :param mean_null: Null hypothesised mean or proportion (float)
    :param mean_alt: Alternative hypothesised mean or proportion (float)
    :param n: Sample size (int)
    :param std: Standard deviation or proportion (float)
    :param is_population_std: Whether the standard deviation is the population (True) or sample (False)
    :param tail: Tailed of the distribution (str), either "two" or "left" or "right"
    :param alpha: Significance level (float), e.g., 0.05 for 5%
    :param stat_type: Type of test (str), either "mean" or "proportion"
    :return: Tuple containing:
        - Confidence Interval (tuple of two floats)
    """
    if stat_type == "mean":
        se = std / math.sqrt(n)  # Standard error
    else:
        # Assuming proportion
        is_valid, validation_message = validate_estimation_inputs(
            mean_null, n)  # Validate inputs
        if not is_valid:
            raise ValueError(validation_message)

        se = math.sqrt((mean_null * (1 - mean_null)) / n)

    if is_population_std or stat_type == "proportion":
        # when σ is known even small sample size but normally distributed
        critical_value_low, critical_value_high = z_distribution(
            mean_null, se, alpha, tail)
        # Type II Error (probability of failing to reject H0 when H0 is false)
        if tail == "two":
            beta = stats.norm.cdf(critical_value_high, loc=mean_alt, scale=se) - \
                stats.norm.cdf(critical_value_low, loc=mean_alt, scale=se)
        elif tail == "left":
            beta = stats.norm.cdf(critical_value_low, loc=mean_alt, scale=se)
        elif tail == "right":
            beta = 1 - stats.norm.cdf(critical_value_high,
                                      loc=mean_alt, scale=se)
    else:
        # The t-distribution `converges` to the z-distribution for large samples
        critical_value_low, critical_value_high = t_distribution(
            mean_null, se, alpha, n, tail)
        # Type II Error (probability of failing to reject H0 when H0 is false)
        df = n - 1
        if tail == "two":
            beta = stats.t.cdf(critical_value_high, df, loc=mean_alt, scale=se) - \
                stats.t.cdf(critical_value_low, df, loc=mean_alt, scale=se)
        elif tail == "left":
            beta = stats.t.cdf(critical_value_low, df, loc=mean_alt, scale=se)
        elif tail == "right":
            beta = 1 - stats.t.cdf(critical_value_high,
                                   df, loc=mean_alt, scale=se)

    return (alpha, beta, 1 - beta)


# Example inputs for mean hypothesis testing (z-test)
mean_null = 30  # Null hypothesis mean
mean_alt = 26  # Alternative hypothesis mean (close to the null mean)
std = 100  # Standard deviation
n = 100  # Sample size for mean
alpha = 0.05  # Significance level (Type I error)

# Perform the calculations for mean using z-test
alpha, beta, power_of_test = calculate_mean_estimation(
    mean_null, mean_alt, n,  std, True,  "left", alpha)

print("Mean Test (Z-Test):")
print(f"Type I Error (Alpha): {alpha:.4f}")
print(f"Type II Error (Beta): {beta:.4f}")
print(f"Power of the Test: {power_of_test:.4f}")

# Example inputs for mean hypothesis testing (t-test)
alpha, beta, power_of_test = calculate_mean_estimation(
    mean_null, mean_alt, n,  std, False,  "two", alpha)

print("\nMean Test (t-Test):")
print(f"Type I Error (Alpha): {alpha:.4f}")
print(f"Type II Error (Beta): {beta:.4f}")
print(f"Power of the Test: {power_of_test:.4f}")

# Example inputs for proportion hypothesis testing
p_null = 0.5  # Null hypothesis proportion
p_alt = 0.65  # Alternative hypothesis proportion (slightly higher)
n = 100  # Sample size for proportion

alpha, beta, power_of_test = calculate_mean_estimation(
    p_null, p_alt, n,  p_null, True,  "two", alpha, 'proportion')

print("\nProportion Test (Z-Test):")
print(f"Type I Error (Alpha): {alpha:.4f}")
print(f"Type II Error (Beta): {beta:.4f}")
print(f"Power of the Test: {power_of_test:.4f}")


""" Math example:
Hypothesis Testing - Type II Error and Power Calculation

Given values:
--------------------
Population standard deviation: σ = 100
Sample size: n = 100
Significance level: α = 0.05
Null hypothesis mean: μ_0 = 30
Alternative population mean: μ = 26

Standard Error Calculation:
--------------------
σ_X = σ / sqrt(n) = 100 / sqrt(100) = 10

Two-tailed Test:
--------------------
1. Critical Z-values: ±1.96
2. Critical X-values:
   - X_critical_lower = μ_0 - (1.96 * σ_X) = 30 - 19.6 = 10.4
   - X_critical_upper = μ_0 + (1.96 * σ_X) = 30 + 19.6 = 49.6
3. Z-scores for Type II Error:
   - Z_β_lower = (10.4 - 26) / 10 = -1.56
   - Z_β_upper = (49.6 - 26) / 10 = 2.36
4. Type II Error Probability (Beta):
   - P(Z_β_lower < Z < Z_β_upper) = P(Z < 2.36) - P(Z < -1.56)
   - = 0.9909 - 0.0598 = 0.9311

Left-tailed Test:
--------------------
1. Critical Z-value: -1.645
2. Critical X-value:
   - X_critical = μ_0 + (-1.645 * σ_X) = 30 - 16.45 = 13.55
3. Z-score for Type II Error:
   - Z_β = (13.55 - 26) / 10 = -1.245
4. Type II Error Probability (Beta):
   - P(Z < -1.245) = 0.1073

Right-tailed Test:
--------------------
1. Critical Z-value: 1.645
2. Critical X-value:
   - X_critical = μ_0 + (1.645 * σ_X) = 30 + 16.45 = 46.45
3. Z-score for Type II Error:
   - Z_β = (46.45 - 26) / 10 = 2.045
4. Type II Error Probability (Beta):
   - P(Z < 2.045) = 0.9793
   - P(Z > 2.045) = 1 - 0.9793 = 0.0207

Final Results:
--------------------
- Two-tailed test Type II error probability (Beta): 0.9311
- Left-tailed test Type II error probability (Beta): 0.1073
- Right-tailed test Type II error probability (Beta): 0.0207
"""
