# Popular Probability Distributions:
#
# | Distribution        |  Type       | When to Use                                           |
# |---------------------|-------------|-------------------------------------------------------|
# | Bernoulli           | Discrete    | Single trial with two outcomes (success/failure).     |
# | Binomial            | Discrete    | Number of successes in a fixed number of trials.      |
# | Poisson             | Discrete    | Number of events in a fixed time/space interval.      |
# | Geometric           | Discrete    | Number of trials until the first success.             |
# | Negative Binomial   | Discrete    | Number of trials until a fixed number of successes.   |
# | Uniform             | Continuous  | All outcomes within a range are equally likely.       |
# | Normal (Gaussian)   | Continuous  | Naturally occurring data, e.g., heights, test scores. |
# | Exponential         | Continuous  | Time between events in a Poisson process.             |
# | Gamma               | Continuous  | Generalized waiting times for events.                 |
# | Beta                | Continuous  | Probabilities constrained between 0 and 1.            |
# | Chi-Square          | Continuous  | Statistical inference, variance estimation.           |
# | Student's t         | Continuous  | Small sample means comparison.                        |
# | Log-Normal          | Continuous  | Skewed distributions in finance, growth rates.        |

"""
Probability Distributions:
    Probability distributions describe how probabilities are assigned to different possible outcomes in a random experiment.
    They are broadly classified into:
        1. Discrete Distributions - Used for countable outcomes (e.g., number of occurrences).
        2. Continuous Distributions - Used for uncountable, continuous outcomes (e.g., measurements).

Terminology:

1. PMF (Probability Mass Function):
   - Defines the probability of a discrete random variable taking a specific value.
   - Example: For a fair die, the PMF assigns a probability of 1/6 to each of the six faces.

2. PDF (Probability Density Function):
   - Describes the likelihood of a random variable taking on a specific value in a continuous distribution.
   - Example: For a normal distribution, the PDF graph is a bell curve.

3. CDF (Cumulative Distribution Function):
   - Represents the probability that a random variable is less than or equal to a specific value.
   - Example: For value `x`, CDF(x) = P(X ≤ x).

4. PPF (Percent Point Function) / Inverse CDF:
   - The inverse of the CDF. It gives the value of the random variable corresponding to a specific cumulative probability.
   - Example: If `p = 0.95`, PPF(0.95) gives the value `x` such that P(X ≤ x) = 0.95.

5. SF (Survival Function):
   - The complement of the CDF, representing the probability that a random variable is greater than a given value.
   - SF(x) = 1 - CDF(x).
   - Example: In reliability analysis, SF(x) gives the probability that a system/component survives beyond time `x`.

Probability Comparisons:

- Probability in Between: P(a ≤ X ≤ b)
  - Calculate the difference between the CDF at `b` and `a`:
    P(a ≤ X ≤ b) = CDF(b) - CDF(a).

- Probability Less Than: P(X < a)
  - Directly use the CDF value at `a`:
    P(X < a) = CDF(a).

- Probability Greater Than: P(X > a)
  - Use the complement of the CDF at `a`:
    P(X > a) = SF(a) = 1 - CDF(a).

Applications:
    - PMF is used for discrete random variables to calculate the probability of specific values.
    - PDF is used to visualize the shape of the distribution and the relative likelihood of values.
    - CDF and probability comparisons help calculate probabilities for given intervals or thresholds.
    - PPF is useful for finding the value of the random variable for a given cumulative probability, often used for
      generating percentiles or for determining critical values in hypothesis testing.
    - SF is useful in survival analysis, reliability engineering, and risk assessment to evaluate the probability
      that a variable exceeds a certain threshold.
"""
