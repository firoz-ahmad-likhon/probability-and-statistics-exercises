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

import math
from scipy import stats
"""Normal Distribution

Problem: The scores of students in a standardized math exam are normally distributed with a mean of 75 and a standard
         deviation of 10.
         Find the probability that a randomly selected student:
            a. Scores exactly 85.
            b. Scores greater than 85 (>85).
            c. Scores less than 65 (<65).
            d. Scores between 70 and 90 (70-90).
            e. Calculate the z-score for a student who scored 95.
"""
mean = 75
std = 10
dist = stats.norm(loc=mean, scale=std)  # Normal distribution
# For a
# Since normal distribution is continuous, the probability of any exact value is zero. P(X = 85) = 0
print("Probability of getting exactly 85:", 0)
# For b
prob = dist.sf(85)
print("Probability of getting greater than 85:", prob)
# For c
prob = dist.cdf(65)
print("Probability of getting less than 65:", prob)
# For d
prob = dist.cdf(90) - dist.cdf(70)
print("Probability of getting between 70 and 90:", prob)
# For e
z_score = (95 - mean) / std
print("Z-score for 95:", z_score)

"""Chi-square Distribution

Problem: A university professor is analyzing the variance in time (in minutes) that students take to complete a
         standardized test. The completion time follows a Chi-Square distribution with 10 degrees of freedom
         (k=10). Find the probability that a randomly selected student:
         a. Takes exactly 15 minutes to complete the test.
         b. Takes more than 15 minutes (>15).
         c. Takes less than 10 minutes (<10).
         d. Takes between 12 and 18 minutes (12-18).
         e. Calculate the mean and standard deviation of the Chi-Square distribution.
"""
k = 10
dist = stats.chi2(df=k)  # Chi-square distribution
# For a
# Since normal distribution is continuous, the probability of any exact value is zero. P(X = 15) = 0
print("Probability of getting exactly 15 minutes:", 0)
# For b
prob = dist.sf(15)
print("Probability of getting more than 15 minutes:", prob)
# For c
prob = dist.cdf(10)
print("Probability of getting less than 10 minutes:", prob)
# For d
prob = dist.cdf(18) - dist.cdf(12)
print("Probability of getting between 12 and 18 minutes:", prob)
# For e
mean = k
std = math.sqrt(2 * k)
print("Mean:", mean)
print("Std:", std)

"""Exponential Distribution

Problem: A laptop manufacturer company XYZ, produces laptops that have an average lifespan of 5 years. The lifespan of
         each laptop follows an exponential distribution. Find the probability that a randomly selected laptop:
         a. Lasts more than 6 years (>6).
         b. Lasts less than 3 years (<3).
         c. Lasts between 4 and 7 years (4-7).
         d. Calculate the mean and standard deviation of the Exponential distribution.
"""
mean = std = 5
rate = 1 / mean     # λ is the inverse of the mean
dist = stats.expon(scale=std)  # Exponential distribution
# For a
prob = dist.sf(6)
print("Probability of lasting more than 6 years:", prob)
# For b
prob = dist.cdf(3)
print("Probability of lasting less than 3 years:", prob)
# For c
prob = dist.cdf(7) - dist.cdf(4)
print("Probability of lasting between 4 and 7 years:", prob)
# For d
print("Mean:", mean)
print("Std:", std)


"""Log-Normal Distribution

Problem: The daily revenue of a small online store follows a log-normal distribution with parameters:
         μ = 4.5 (mean of log-transformed revenue) and σ = 0.954 (standard deviation of log-transformed revenue)
         Calculate the following probabilities:
         a. P(X > 200) → Probability of daily revenue greater than $200
         b. P(X < 100) → Probability of daily revenue less than $100
         c. P(100 ≤ X ≤ 200) → Probability that daily revenue falls between $100 and $200
"""
mean = 4.5  # Mean of log(X)
std = 0.954  # Standard deviation of log(X)
dist = stats.lognorm(s=std, scale=math.exp(mean))  # log-normal distribution
# For a
prob = dist.sf(200)  # 1 - CDF(200)
print("Probability of revenue greater than $200:", prob)
# For b
prob = dist.cdf(100)
print("Probability of revenue less than $100:", prob)
# For c
prob = dist.cdf(200) - dist.cdf(100)
print("Probability of revenue between $100 and $200:", prob)


"""Binomial Distribution.

Problem: A multiple choice test contains 20 questions with answer choice A, B, C and D. Only one answer choice to
         each question represents a correct answer. Find the probability that a student will answer
         a. exactly 6 questions correct
         b. at least 6 questions correct (>6)
         c. at most 6 questions correct (<6)
         d. between 4 and 8 questions correct (4-8)
         if he makes random guesses on all 20 questions.
         e. calculate the mean and std deviation of the binomial distribution.
"""
n = 20
p = 1 / 4  # Probability of getting a correct answer
dist = stats.binom(n=n, p=p)  # Binomial distribution
# For a
k = 6
prob = dist.pmf(k)
print("Probability of getting 6 questions correct:", prob)
# For b
k = 5
prob = dist.sf(k)
print("Probability of getting at least 6 questions correct:", prob)
# For c
k = 6
prob = dist.cdf(k)
print("Probability of getting at most 6 questions correct:", prob)
# For d
k1, k2 = 4, 8
prob = dist.cdf(k2) - dist.cdf(k1)
print("Probability of getting between 4 and 8 questions correct:", prob)
# For e
mean = n * p  # or dist.mean()
std = math.sqrt(n * p * (1 - p))  # or dist.std()
print("Mean:", mean)
print("Std:", std)

"""Geometric Distribution.

Problem: 2% of all tires produced by company XYZ has a defect. A random sample of 100 tires is tested for quality
        assurance. Find the probability that:
         a. The 7th tire selected is the first defective one.          → P(X = 7)
         b. The first defective tire appears after selecting at least 10 tires. → P(X ≥ 10)
         c. The first defective tire appears within the first 5 tires. → P(X ≤ 5)
         d. The first defective tire appears between the 8th and 15th selected tires (inclusive). → P(8 ≤ X ≤ 15)
         e. How many tires would you expect to test until finding the first defective one? → E(X)
         f. Calculate the standard deviation of the geometric distribution. → SD(X)
"""
p = 0.02  # Probability of defect
dist = stats.geom(p=p)  # Geometric distribution
# For a: P(X = 7)
k = 7
prob = dist.pmf(k)
print("Probability of getting 7th tire as defective:", prob)
# For b: P(X ≥ 10)
k = 10
prob = dist.sf(k)
print("Probability of getting defective tire after selecting at least 10 tires:", prob)
# For c: P(X ≤ 5)
k = 5
prob = dist.cdf(k)
print("Probability of getting defective tire within the first 5 tires:", prob)
# For d: P(8 ≤ X ≤ 15)
k1, k2 = 8, 15
prob = dist.cdf(k2) - dist.cdf(k1)
print("Probability of getting defective tire between 8th and 15th selected tires:", prob)
# For e: E(X)
mean = 1 / p
print("Expected number of tires to test until finding the first defective one:", mean)
# For f
std = math.sqrt((1 - p) / p**2)
print("Std:", std)

"""Poisson Distribution.

Problem: A student receives an average of 7 text messages over a 2-hour period. The number of text messages
         received follows a Poisson distribution. Find the probability that the student will receive:
         a. exactly 9 text messages in 2 hours
         b. at least 9 text messages (> 9) in 2 hours
         c. at most 9 text messages (< 9) in 2 hours
         d. between 4 and 8 text messages (4 to 8) in 2 hours
         e. calculate the mean and standard deviation of the Poisson distribution
         f. exactly 60 test messages in 24 hours.
"""
mean = 7  # mean = λ
dist = stats.poisson(mu=mean)  # Poisson distribution
# For a
k = 9
prob = dist.pmf(k)
print("Probability of getting 9 text in 2 hours:", prob)
# For b
k = 9
prob = dist.sf(k)
print("Probability of getting at least 9 text in 2 hours:", prob)
# For c
k = 9
prob = dist.cdf(k)
print("Probability of getting at most 9 text in 2 hours:", prob)
# For d
k1, k2 = 4, 8
prob = dist.cdf(k2) - dist.cdf(k1)
print("Probability of getting between 4 and 8 text  in 2 hours:", prob)
# For e
std = math.sqrt(mean)  # or dist.std()
print("Mean:", mean)
print("Std:", std)
# For f
k = 60
mean_24 = (mean * 24) / 2  # Average number of text messages for 24 hours
prob = stats.poisson.pmf(k, mu=mean_24)
print("Probability of getting 60 text in 24 hours:", prob)
