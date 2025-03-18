"""
Chi-Square (χ²) Test:
    The Chi-Square (χ²) test is a statistical hypothesis test used to determine if there is a significant
    association between categorical variables or if an observed distribution of categorical data differs from an expected theoretical distribution.
    Equation:
        χ² = Σ( (Oᵢ - Eᵢ)² / Eᵢ )
    Where:
        O = Observed frequency
        E = Expected frequency

Types of Chi-Square (χ²) Tests:
    1. Chi-Square (χ²) Goodness-of-Fit Test:
        - Determines if the observed distribution of a single categorical variable matches an expected theoretical distribution.
        - Hypotheses:
            - H₀: The observed data follows the expected distribution.
            - H₁: The observed data does not follow the expected distribution.
        - Example: Does the distribution of candy colors in a sample match the company's claim that each color appears equally?
        Expectation Calculation:
            The expected frequency for each category is:
                Eᵢ = (Total observations) / (Number of categories)
            If a specific distribution is assumed (e.g., uniform, binomial), Eᵢ is calculated based on that distribution's expected probabilities.

    2. Chi-Square (χ²) Test for Independence/Association:
        - Tests whether there is a significant association between two categorical variables in a sample.
        - Uses contingency tables to compare observed and expected frequencies.
        - Hypotheses:
            - H₀: The two categorical variables are independent (no association).
            - H₁: The two categorical variables are dependent (there is an association).
        - Example: Is there a relationship between smoking status (smoker/non-smoker) and lung disease (yes/no)?

        Expectation Calculation:
            The expected frequency for each cell in a contingency table is calculated by:
                Eᵢ = (Row total for category i * Column total for category j) / (Total observations)

    3. Chi-Square (χ²) Test for Homogeneity:
        - Compares the distribution of a categorical variable across multiple independent groups.
        - Determines if different populations have the same proportion of a certain characteristic.
        - Hypotheses:
            - H₀: The distributions of the categorical variable are the same across groups.
            - H₁: At least one group's distribution is different.
        - Example: Do different brands of a product (Brand A, Brand B, Brand C) have the same proportion of customer satisfaction levels?

        Expectation Calculation:
            Like the test for independence, the expected frequency for each group is:
                Eᵢ = (Row total for category i * Column total for category j) / (Total observations)

Estimating Population Variance:
    The Chi-Square distribution is used to estimate the population variance (σ²) when the sample variance (s²) is known.
    Given a random sample of size n from a normal population, the Chi-Square statistic for variance is:
        χ² = ( (n - 1) * s² ) / σ²
        σ² = ( (n - 1) * s² ) / χ²

    Confidence Interval for Variance:
        ( (n−1) × s² / χ²₍ₐ/₂₎, (n−1) × s² / χ²₍₁−ₐ/₂₎ )
    Where:
        n = Sample size
        s² = Sample variance
        σ² = Population variance (unknown)
        α is the significance level
        χ²₍ₐ/₂₎ and χ²₍₁−ₐ/₂₎ are critical values from the Chi-Square distribution

P-value:
    - The probability of observing a Chi-Square statistic as extreme as the one calculated, assuming the null hypothesis is true.
    - Decision Rule:
        - If P-value ≤ α (significance level), reject the null hypothesis (there is a significant difference/association/variance).
        - If P-value > α, fail to reject the null hypothesis (no significant difference/association/variance).

Other Statistical Properties:
    - Degrees of Freedom (df):
        - Goodness-of-fit: df = (number of categories - 1)
        - Independence & Homogeneity: df = (rows - 1) * (columns - 1)
    - Mean: E(χ²) = df
    - Median (Approximate): df - (2/3)
    - Variance: Var(χ²) = 2df
    - Standard Deviation: std(χ²) = √(2df)

Assumptions of Chi-Square Tests:
    - Data should consist of independent observations.
    - The sample data must be categorical (nominal or ordinal) and mutually exclusive.
    - Expected frequency in each category should be at least 5 for the test to be valid.

Limitations:
    - Chi-Square tests may not be accurate if expected frequencies are small (less than 5). In such cases, Fisher's exact test or other tests may be more appropriate.

Chi-Square (χ²) Convergence to Normal Distribution:
    For large degrees of freedom (df → ∞), the Chi-Square distribution approaches a normal distribution: χ²(df) ≈ N(df, 2df)

Application Areas:
    - Genetics (Hardy-Weinberg equilibrium testing)
    - Market research (consumer preference analysis)
    - Medical research (disease association studies)
    - Quality control (defect rate analysis)
    - Social sciences (survey response analysis)
    - Finance (risk management and portfolio analysis)
    - Education (analysis of student performance across different groups)
"""

from scipy.stats import chi2
import numpy as np


def validate_expected_frequency(expected: np.ndarray) -> None:
    """Validates expected frequencies.

    :param expected: Expected frequency.
    :raises ValueError: If expected frequency is less than 5.
    """
    if np.any(expected < 5):
        raise ValueError(
            "Error: Expected frequency is less than 5, which may violate assumptions.")


"""Goodness-of-Fit Test.

Problem:
    A company claims that its candy colors are equally distributed among four colors.
    To verify this claim, a sample of 100 candies is collected with the following observed frequencies:

    | Red | Green | Blue | Yellow |
    |-----|-------|------|--------|
    | 20  |  30   |  25  |   25   |

    Test whether the observed distribution of colors follows the company's claim.

Hypotheses:
    - H₀ (Null Hypothesis): The observed data follows the expected distribution.
    - H₁ (Alternative Hypothesis): The observed data does not follow the expected distribution.
"""

actual = np.array([20, 30, 25, 25])
n = actual.sum()  # Total number of candies
length = len(actual)  # Number of colors
expected = n / length * np.ones(length)  # Expected frequency for each color

validate_expected_frequency(expected)  # Validate expected frequencies

df = length - 1  # Degree of freedom
# Chi-Squared statistic calculation
chi_squared = np.sum((actual - expected) ** 2 / expected)

dist = chi2(df=df)  # Chi-Square distribution
p = dist.sf(chi_squared)  # 1 - dist.cdf(chi_squared)

alpha = .05  # Significance level
print(chi_squared, p, p <= alpha)

"""Independence/Association Test.

Problem:
    A researcher wants to determine whether there is an association between
    smoking status (Smoker, Non-Smoker) and lung disease (Yes, No). The collected
    data is presented in the following contingency table:

        | Smoking Status | Lung Disease (Yes) | Lung Disease (No) |
        |----------------|--------------------|-------------------|
        | Smoker         | 40                 | 60                |
        | Non-Smoker     | 30                 | 170               |

Hypotheses:
    - H₀ (Null Hypothesis): Smoking and lung disease are independent (no association).
    - H₁ (Alternative Hypothesis): Smoking and lung disease are associated.
"""
actual = np.array([
    [40, 60],  # Smoker
    [30, 170],  # Non-Smoker
])
# Calculate row and column totals
row_totals = actual.sum(axis=1, keepdims=True)
col_totals = actual.sum(axis=0, keepdims=True)
total_observations = actual.sum()
expected = np.outer(row_totals, col_totals) / \
    total_observations  # Expected frequency for each cell

validate_expected_frequency(expected)  # Validate expected frequencies

# Chi-Squared test statistic calculation
chi_squared = np.sum((actual - expected) ** 2 / expected)
rows, columns = actual.shape
df = (rows - 1) * (columns - 1)  # Degrees of freedom

dist = chi2(df=df)  # Chi-Square distribution
p = dist.sf(chi_squared)  # 1 - dist.cdf(chi_squared)

alpha = .05  # Significance level
print(chi_squared, p, p <= alpha)

"""Homogeneity Test.

Problem:
    A researcher wants to determine whether the preference for different types of beverages
    (Tea, Coffee, Juice) is the same across three different cities (City A, City B, City C).
    The collected data is as follows:

        | Beverage | City A | City B | City C |
        |----------|--------|--------|--------|
        | Tea      | 50     | 30     | 20     |
        | Coffee   | 40     | 50     | 60     |
        | Juice    | 30     | 40     | 50     |

Hypotheses:
    - H₀ (Null Hypothesis): The distribution of beverage preference is the same across all cities.
    - H₁ (Alternative Hypothesis): The distribution of beverage preference is different across cities.
"""
actual = np.array([
    [50, 30, 20],  # Tea
    [40, 50, 60],  # Coffee
    [30, 40, 50],   # Juice
])
# Compute row and column totals
row_totals = actual.sum(axis=1)
col_totals = actual.sum(axis=0)
total_observations = actual.sum()
expected = np.outer(row_totals, col_totals) / \
    total_observations  # Expected frequency for each cell

validate_expected_frequency(expected)  # Validate expected frequencies

# Chi-Square Test Statistic
chi_squared = np.sum((actual - expected) ** 2 / expected)

rows, columns = actual.shape
df = (rows - 1) * (columns - 1)  # Degrees of freedom
p = chi2(df=df).sf(chi_squared)  # Survival function (1 - CDF)

alpha = .05  # Significance level
print(chi_squared, p, p <= alpha)
