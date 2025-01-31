"""
Estimation:
   Estimation is the process of inferring the value of an unknown population parameter (e.g., mean, proportion) based on sample data.

Equations:
    Interval Estimation = Point Estimation ± Margin of Error
    Margin of Error = Critical Value * Standard Error
        - Critical Value = Z or t
        - Standard Error:
            - For Z-Distribution: σ / √n
            - For t-Distribution: s / √n

Margin of Error (ME):
   - A measure of the precision of an estimate, defining the range within which the true population parameter is likely to lie.
   - ME is influenced by:
     - Confidence level (e.g., 95%, 99%): Higher confidence levels increase the ME.
     - Sample size (n): Larger sample sizes decrease the ME.
     - Variability in data: Greater variability increases the ME.

Confidence Level:
    - The confidence level represents the percentage of times the true population parameter is expected to fall within the confidence interval if the sampling process is repeated many times. It is typically expressed as a percentage, such as 90%, 95%, or 99%.

Significant Level (α):
   - The probability of rejecting the null hypothesis when it is true (Type I error).
   - Formula for two-tailed test:
     - α = (1 - Confidence Level) / 2
     - For example, for a 95% confidence level: α = (1 - 0.95) / 2 = 0.025

Critical Value:
    - Defines the boundary for confidence intervals or hypothesis tests.
    - Z-Critical: Z = InverseCDF(1 - α/2) from normal distribution
    - t-Critical: t = InverseCDF(1 - α/2) from t-distribution

Types of Estimation:
   1. Point Estimation:
      - A single value is used to estimate the parameter of interest (e.g., sample mean x̄ as an estimate of the population mean μ).
   2. Interval Estimation:
      - Provides a range of values (e.g., confidence interval) within which the parameter is likely to lie. This range includes an associated confidence level, which indicates the probability that the interval contains the true population parameter.

Estimation Methods:
    1. Mean Estimation:
    a) Z-Distribution (Used when sample size (n) is large (n ≥ 30) or the population standard deviation (σ) is known):
       - Equation: x̄ ± Z * (σ / √n)
         where Z is the critical value from the Z-distribution table.
    b) t-Distribution (Used when sample size (n) is small (n < 30) and σ is unknown):
       - Equation: x̄ ± t * (s / √n)
         where s is the sample standard deviation and t is the critical value from the t-distribution table.

    2. Proportion Estimation (Used for binary data (e.g., success/failure)):
    - Equation: p̂ ± Z * √(p̂(1 - p̂) / n), where p̂ is the sample proportion.
    - Validation of Proportion:
         - n * p̂ ≥ 10 (expected successes)
         - n * (1 - p̂) ≥ 10 (expected failures)
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


def z_distribution(mean: float, se: float, cl: float, range_start: float | None, range_end: float | None) -> tuple[np.float64, tuple[np.float64, np.float64], np.float64 | None]:
    """Estimate using Z Distribution.

    :param mean: Sample Mean (float)
    :param se: Standard Error (float)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param range_start: Start of the range for probability calculation (float or None)
    :param range_end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    alpha = (1 - cl) / 2  # Significance level
    z_critical = stats.norm.ppf(1 - alpha)
    moe = z_critical * se  # Margin of Error
    ci = (mean - moe, mean + moe)  # Confidence Interval

    probability = None
    if range_start is not None and range_end is not None:
        '''The formula for calculating z score without generating a distribution object:
        z_start = (range_start - mean) / se
        z_end = (range_end - mean) / se
        probability_within_range = stats.norm.cdf(z_end) - stats.norm.cdf(z_start)'''
        dist = stats.norm(mean, se)  # Create normal distribution object
        probability = dist.cdf(range_end) - dist.cdf(range_start)

    return moe, ci, probability


def t_distribution(mean: float, se: float, cl: float, n: int, range_start: float | None, range_end: float | None) -> tuple[np.float64, tuple[np.float64, np.float64], np.float64 | None]:
    """Estimate using T Distribution.

    :param mean: Sample mean (float)
    :param se: Standard Error (float)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param n: Sample size (int)
    :param range_start: Start of the range for probability calculation (float or None)
    :param range_end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    df = n - 1  # Degree of Freedom
    alpha = (1 - cl) / 2  # Significance level
    t_critical = stats.t.ppf(1 - alpha, df)
    moe = t_critical * se  # Margin of Error
    ci = (mean - moe, mean + moe)  # Confidence Interval

    probability = None
    if range_start is not None and range_end is not None:
        '''The formula for calculating t score without generating a distribution object:
        t_start = (range_start - mean) / se
        t_end = (range_end - mean) / se
        probability_within_range = stats.t.cdf(t_end, df) - stats.t.cdf(t_start, df)'''
        dist = stats.t(df, mean, se)
        probability = dist.cdf(range_end) - dist.cdf(range_start)

    return moe, ci, probability


def calculate_mean_estimation(mean: float, n: int, std: float, is_population_std: bool = True, cl: float = 0.95, start: float | None = None, end: float | None = None) -> tuple[np.float64, tuple[np.float64, np.float64], np.float64 | None]:
    """Estimate population parameters using the mean and standard deviation.

    :param mean: Sample mean (float)
    :param std: Standard deviation (float)
    :param is_population_std: Whether the standard deviation is the population (True) or sample (False)
    :param n: Sample size (int)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param start: Start of the range for probability calculation (float or None)
    :param end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the value falling within the specified range (float or None)
    """
    se = std / math.sqrt(n)  # Standard error

    if is_population_std:
        # when σ is known even small sample size but normally distributed
        return z_distribution(mean, se, cl, start, end)
    else:
        # The t-distribution `converges` to the z-distribution for large samples
        return t_distribution(mean, se, cl, n, start, end)


def calculate_proportion_estimation(p: float, n: int, cl: float = 0.95, range_start: float | None = None, range_end: float | None = None) -> tuple[np.float64, tuple[np.float64, np.float64], np.float64 | None]:
    """Estimate population parameters using proportion.

    :param p: Sample proportion (float), where 0 <= p <= 1
    :param n: Sample size (int)
    :param cl: Confidence level (float), e.g., 0.95 for 95% confidence
    :param range_start: Start of the range for probability calculation (float or None)
    :param range_end: End of the range for probability calculation (float or None)
    :return: Tuple containing:
        - Margin of Error (float)
        - Confidence Interval (tuple of two floats)
        - Probability of the proportion falling within the specified range (float or None)
    """
    # Validate inputs
    is_valid, validation_message = validate_estimation_inputs(p, n)
    if not is_valid:
        raise ValueError(validation_message)

    se = math.sqrt((p * (1 - p)) / n)

    return z_distribution(p, se, cl, range_start, range_end)


# Example usage for mean estimation
sample_mean = 2.29
sample_std = .20
sample_size = 12
confidence_interval = 0.90
start = 2.1
end = 2.25

moe, ci, probability = calculate_mean_estimation(
    sample_mean, sample_size, sample_std, False, confidence_interval, start, end)
print(f"Mean Estimation (t-distribution) - Margin of Error: {moe:.2f}")
print(
    f"Mean Estimation (t-distribution) - Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
print(
    f"Mean Estimation (t-distribution) - Probability within range [{start}, {end}]: {probability:.4f}")
print("\n")

sample_mean = 299720
population_std = 68650
sample_size = 1500
confidence_interval = 0.95
start = 290000
end = 300000

moe, ci, probability = calculate_mean_estimation(
    sample_mean, sample_size, population_std, True, confidence_interval, start, end)
print(f"Mean Estimation (z-distribution) - Margin of Error: {moe:.2f}")
print(
    f"Mean Estimation (z-distribution) - Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
print(
    f"Mean Estimation (z-distribution) - Probability within range [{start}, {end}]: {probability:.4f}")
print("\n")

# Proportion estimation
sample_proportion = 0.44
sample_size = 1000
confidence_interval = 0.95
start = 0.45
end = 0.47

try:
    moe, ci, probability = calculate_proportion_estimation(
        sample_proportion, sample_size, confidence_interval, start, end)
    print(f"Proportion Estimation - Margin of Error: {moe:.4f}")
    print(
        f"Proportion Estimation - Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")
    print(
        f"Proportion Estimation - Probability within range [{start}, {end}]: {probability:.4f}")
except ValueError as e:
    print(f"Validation Error: {e}")
