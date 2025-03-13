# Popular Probability Distributions:
#
# Popular Probability Distributions:

# | Distribution      | Type       | When to Use                                           | Parameters                             |
# |-------------------|------------|-------------------------------------------------------|----------------------------------------|
# | Bernoulli         | Discrete   | Single trial with two outcomes (success/failure).     | p (probability of success)             |
# | Binomial          | Discrete   | Number of successes in a fixed number of trials.      | n (trials), p (success probability)    |
# | Poisson           | Discrete   | Number of events in a fixed time/space interval.      | λ (average event rate)                 |
# | Geometric         | Discrete   | Number of trials until the first success.             | p (success probability)                |
# | Negative Binomial | Discrete   | Number of trials until a fixed number of successes.   | r (successes), p (success probability) |
# | Uniform           | Continuous | All outcomes within a range are equally likely.       | a (min), b (max)                       |
# | Normal            | Continuous | Naturally occurring data, e.g., heights, test scores. | μ (mean), σ (std. deviation)           |
# | Exponential       | Continuous | Time between events in a Poisson process.             | λ (rate parameter)                     |
# | Gamma             | Continuous | Generalized waiting times for events.                 | k (shape), θ (scale)                   |
# | Beta              | Continuous | Probabilities constrained between 0 and 1.            | α (shape1), β (shape2)                 |
# | Chi-Square        | Continuous | Statistical inference, variance estimation.           | k (degrees of freedom)                 |
# | Student's t       | Continuous | Small sample means comparison.                        | ν (degrees of freedom)                 |
# | Log-Normal        | Continuous | Skewed distributions in finance, growth rates.        | μ (mean of log), σ (std. of log)       |


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

Memoryless Property:
    The memoryless property refers to a characteristic of certain types of random variables, where the probability of
    an event occurring in the future is independent of the past. In other words, given that the random event has not
    yet occurred, the probability of it happening in the next time interval is the same, regardless of how long it has
    been since the event started.

    This property is most commonly associated with the exponential distribution and the geometric distribution.

    Example:
        In a real-life scenario, consider a bus that arrives at a bus stop at random intervals, and you are waiting for
        the bus. If the bus has not arrived yet, the memoryless property implies that your expectation of the time
        until the next bus arrives is unaffected by how long you’ve already been waiting.

        Mathematically, if the time until the bus arrives follows an exponential distribution, the probability of the
        bus arriving in the next t minutes is the same, regardless of how long you’ve already been waiting.

    Formally, the memoryless property can be expressed as:
        P(T > t + s | T > t) = P(T > s)
        where T is a random variable representing the time until the event occurs,
        t and s are time intervals, and P represents the probability.
"""
