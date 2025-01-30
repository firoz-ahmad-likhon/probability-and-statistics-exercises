# Hypothesis condition conversion:
#
# | Condition          | H₀    | H₁    | Test Type    |
# |--------------------|-------|-------|--------------|
# | Equal              | =     | ≠     | Two-tailed   |
# | Greater than       | ≥     | <     | Left-tailed  |
# | Less than          | ≤     | >     | Right-tailed |

"""
Hypothesis Testing:
   - Hypothesis testing is a statistical method used to make inferences or draw conclusions about a population parameter based on sample data.

Null Hypothesis (H₀):
      - Represents the default assumption or status quo; no effect or no difference is assumed.
      - Example: "The mean test score of students is 75." (H₀: μ = 75)

Alternative Hypothesis (H₁ or Ha):
      - Represents the claim to be tested, contrary to the null hypothesis.
      - Example: "The mean test score of students is not 75." (H₁: μ ≠ 75)

Significance Level (α):
   - The probability of rejecting the null hypothesis when it is true (Type I error).
   - Commonly used levels: α = 0.05 (5%), α = 0.01 (1%).
   - For two-tailed tests, the significance level is split equally across both tails: α/2 in each tail.

Critical Value:
   - The threshold value(s) that define the boundary of the rejection region (critical region) in a hypothesis test.
   - For Z-tests and T-tests, these values are based on the significance level (α) and the test type:
       - One-Tailed Test: Critical value lies in one tail.
       - Two-Tailed Test: Critical values lie in both tails, splitting α/2 in each tail.
   - Example (Two-tailed Z-test, α = 0.05): Critical values are ±1.96.

P-Value:
   - The probability of observing a sample statistic as extreme as (or more extreme than) the observed value, assuming H₀ is true.
   - Decision Rule:
       - If P-Value ≤ α, reject H₀ (evidence supports H₁).
       - If P-Value > α, fail to reject H₀ (insufficient evidence to support H₁).

Test Types:
   - One-Tailed Test:
       - Used when the alternative hypothesis specifies a direction.
       - Example (Right-Tailed): H₁: μ > μ₀ (tests if the population mean is greater than μ₀).
       - Example (Left-Tailed): H₁: μ < μ₀ (tests if the population mean is less than μ₀).

   - Two-Tailed Test:
       - Used when the alternative hypothesis does not specify a direction (tests for any difference).
       - Example: H₁: μ ≠ μ₀ (tests if the population mean is different from μ₀).
       - The rejection region is in both tails of the distribution.
"""
from typing import NoReturn, cast

from scipy import stats
import math
import numpy as np


def validate_proportion_inputs(p: float, n: int) -> tuple[bool, str]:
    """Validate inputs for proportion hypothesis testing.

    :param p: Sample proportion (float)
    :param n: Sample size (int)
    :return: Tuple of boolean and validation message
    """
    if n * p < 10 or n * (1 - p) < 10:
        return False, "For proportion estimation, succeeded = np and failed = n(1-p) must both be at least 10."
    return True, "Inputs are valid."


def calculate_standard_error_mean(std: float, n: int) -> float:
    """Calculate the standard error for the mean.

    :param std: Standard deviation (float)
    :param n: Sample size (int)
    :return: Standard error (float)
    """
    return std / math.sqrt(n)


def calculate_standard_error_proportion(p: float, n: int) -> float:
    """Calculate the standard error for a proportion.

    :param p: Sample proportion (float)
    :param n: Sample size (int)
    :return: Standard error (float)
    """
    return math.sqrt((p * (1 - p)) / n)


def calculate_z_score(sample_stat: float, population_mean: float, se: float) -> float:
    """Calculate the z-score for a hypothesis test.

    :param sample_stat: Sample statistic (mean or proportion) (float)
    :param population_mean: Hypothesized population mean or proportion (float)
    :param se: Standard error (float)
    :return: Z-score (float)
    """
    return (sample_stat - population_mean) / se


def calculate_t_score(sample_mean: float, population_mean: float, se: float) -> float:
    """Calculate the t-score for a hypothesis test.

    :param sample_mean: Sample mean (float)
    :param population_mean: Hypothesized population mean (float)
    :param se: Standard error (float)
    :return: T-score (float)
    """
    return (sample_mean - population_mean) / se


def calculate_p_value(z_score: float, tail_type: str = 'two', df:  int | None = None) -> np.float64 | NoReturn:
    """Calculate the p-value for a hypothesis test.

    :param z_score: Z-score or T-score (float)
    :param tail_type: Type of test ('two', 'left', 'right')
    :param df: Degrees of freedom (int), used for T-distribution
    :return: p-value (float)
    """
    if tail_type == 'two':
        if df is None:
            """Calculation strategy:
            testing on left tail: distribution.cdf(-abs(z_score)
            testing on right tail: 1 - distribution.cdf(abs(z_score))
            """
            return cast(np.float64, 2 * (1 - stats.norm.cdf(abs(z_score))))
        else:
            return cast(np.float64, 2 * (1 - stats.t.cdf(abs(z_score), df)))
    elif tail_type == 'left':
        if df is None:
            return cast(np.float64, stats.norm.cdf(z_score))
        else:
            return cast(np.float64, stats.t.cdf(z_score, df))
    elif tail_type == 'right':
        if df is None:
            return cast(np.float64, 1 - stats.norm.cdf(z_score))
        else:
            return cast(np.float64, 1 - stats.t.cdf(z_score, df))
    else:
        raise ValueError(
            "Invalid tail type. Choose 'two', 'left', or 'right'.")


def hypothesis_test_mean(sample_mean: float, population_mean: float, std: float, n: int, alpha: float, tail_type: str = 'two', is_population_std: bool = True) -> tuple[float, np.float64, str]:
    """Perform a hypothesis test for the mean.

    :param sample_mean: Sample mean (float)
    :param population_mean: Hypothesized population mean (float)
    :param std: Population or Sample standard deviation (float)
    :param n: Sample size (int)
    :param alpha: Significance level (float)
    :param tail_type: Type of test ('two', 'left', 'right')
    :param is_population_std: Whether the standard deviation is the population (True) or sample (False)
    :return: Tuple containing the test statistic (Z or T), p-value, and decision
    """
    se = calculate_standard_error_mean(std, n)

    if is_population_std:
        # when σ is known even small sample size but normally distributed
        test_stat = calculate_z_score(sample_mean, population_mean, se)
        p_value = calculate_p_value(test_stat, tail_type)
    else:
        # The t-distribution `converges` to the z-distribution for large samples
        test_stat = calculate_t_score(sample_mean, population_mean, se)
        df = n - 1  # Degrees of freedom
        p_value = calculate_p_value(test_stat, tail_type, df)

    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"
    return test_stat, p_value, decision


def hypothesis_test_proportion(sample_proportion: float, population_proportion: float, n: int, alpha: float, tail_type: str = 'two') -> tuple[float, np.float64, str]:
    """Perform a hypothesis test for the proportion.

    :param sample_proportion: Sample proportion (float)
    :param population_proportion: Hypothesized population proportion (float)
    :param n: Sample size (int)
    :param alpha: Significance level (float)
    :param tail_type: Type of test ('two', 'left', 'right')
    :return: Tuple containing the z-score, p-value, and decision
    """
    is_valid, validation_message = validate_proportion_inputs(
        population_proportion, n)
    if not is_valid:
        raise ValueError(validation_message)

    se = calculate_standard_error_proportion(population_proportion, n)
    z_score = calculate_z_score(sample_proportion, population_proportion, se)
    p_value = calculate_p_value(z_score, tail_type)
    decision = "Reject H₀" if p_value < alpha else "Fail to Reject H₀"

    return z_score, p_value, decision


# Example Usage
sample_mean = 17  # Sample mean
population_mean = 15  # Hypothesized population mean
std = 0.5  # Standard deviation (known population std)
n = 10  # Sample size
alpha = 0.05  # Significance level

# Z-Test for Mean
test_stat, p_value, decision = hypothesis_test_mean(
    sample_mean, population_mean, std, n, alpha, 'right', is_population_std=True)
print(
    f"Mean Test (Z-Test) - Test Statistic: {test_stat:.2f}, p-value: {p_value:.4f}, Decision: {decision}")

# T-Test for Mean
test_stat, p_value, decision = hypothesis_test_mean(
    sample_mean, population_mean, std, n, alpha, 'right', is_population_std=False)
print(
    f"Mean Test (T-Test) - Test Statistic: {test_stat:.2f}, p-value: {p_value:.4f}, Decision: {decision}")

# Two-tailed Z-Test for Mean
sample_mean = 350  # Sample mean
population_mean = 355  # Hypothesized population mean
std = 8  # Standard deviation (known population std)
n = 30  # Sample size

# Z-Test for Mean
test_stat, p_value, decision = hypothesis_test_mean(sample_mean, population_mean, std, n, alpha, 'two',
                                                    is_population_std=True)
print(
    f"Two tail Mean Test (Z-Test) - Test Statistic: {test_stat:.2f}, p-value: {p_value:.4f}, Decision: {decision}")

# Z-Test for Proportion
sample_proportion = 0.44  # Sample proportion
population_proportion = 0.5  # Hypothesized population proportion
n = 1000  # Sample size
alpha = 0.05  # Significance level

z_score, p_value, decision = hypothesis_test_proportion(
    sample_proportion, population_proportion, n, alpha, 'two')
print(
    f"Proportion Test - Z-score: {z_score:.2f}, p-value: {p_value:.4f}, Decision: {decision}")
