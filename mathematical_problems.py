import math
from scipy import stats

"""Estimating Confidence Interval.

Problem: A researcher wants to estimate the confidence interval for the average sales price of medium-sized apartments
         in a city.For this, the researcher collected a sample of 1,500 apartments, and the sample's mean price was
         $299,720. It is known that the population standard deviation is $68,650. Find the 95% confidence interval for
         the average sales price of apartments in the city.
"""
n = 1500
x = 299720  # Sample mean
std = 68650  # Population standard deviation
a = .05  # Significance level (5%)
mean = x
se = std / math.sqrt(n)  # Standard error
z_critical = stats.norm.ppf(1 - a / 2)  # Z-critical value for two-tailed test
moe = z_critical * se  # Margin of error
cl = (mean - moe, mean + moe)  # Confidence interval
print(cl)

"""Estimating Confidence Interval.

Problem: A doctor wants to estimate the confidence interval for the average cholesterol level of adult men in a city.
         A sample of 25 men gives an average cholesterol level of 186 mg/dL, and the sample's standard deviation is
         12 mg/dL. Find the 95% confidence interval for the average cholesterol level of adult men in the city.
"""
n = 25
x = 186  # Sample mean
s = 12  # Sample standard deviation
a = .05  # Significance level (5%)
mean = x
se = s / math.sqrt(n)  # Standard error
# T-critical value for two-tailed test
t_critical = stats.t.ppf(1 - a / 2, n - 1)
moe = t_critical * se  # Margin of error
cl = (mean - moe, mean + moe)  # Confidence interval
print(cl)

"""Estimating Confidence Interval for Proportion.

Problem: In 2009, the Pew Research Center conducted a survey asking people if religion played an important role in
         their lives. The sample of 1,000 people found that 44% responded affirmatively. Find the 99% confidence
         interval for the proportion of people in the population who consider religion important.
"""
n = 1000
p = .44  # Sample proportion
a = .01  # Significance level (1%)
# Check if conditions for normal approximation are met
print(p * n >= 10 and (1 - p) * n >= 10)
se = math.sqrt((p * (1 - p)) / n)  # Standard error for proportion
z_critical = stats.norm.ppf(1 - a / 2)  # Z-critical value for two-tailed test
moe = z_critical * se  # Margin of error
cl = (p - moe, p + moe)  # Confidence interval
print(cl)

"""Hypothesis Test Two-tailed.

Problem: A company claims that their employees take an average of 90 minutes to learn to use a new machine.
         A sample of 20 employees has an average time of 85 minutes, with a standard deviation of 7 minutes.
         Test the claim at the 1% significance level (99% confidence).

Hypotheses:
    - H₀: The average time for employees to learn to use the new machine is 90 minutes. μ = 90
    - H₁: The average time for employees to learn to use the new machine is not 90 minutes.  μ ≠ 90
"""
mean = 90  # Claimed mean
n = 20
x = 85  # Sample mean
std = 7  # Sample standard deviation
a = .01  # Significance level (1%)
se = std / math.sqrt(n)  # Standard error
dist = stats.norm(mean, se)  # Normal distribution under null hypothesis
p = 2 * dist.cdf(x)  # Two-tailed p-value
print(p, p <= a)  # Accept or reject the null hypothesis

""" Hypothesis Test Left-tailed.

Problem: A mayor claims that the average family wealth in his city is at least $300,000. A sample of 25 families shows
         an average wealth of $288,000, with a population standard deviation of $80,000. Test the mayor's claim at the
         2.5% significance level.

Hypotheses:
    - H₀: The average family wealth in the city is at least $300,000. μ >= 300,000
    - H₁: The average family wealth in the city is less than $300,000. μ < 300,000
"""
mean = 300000  # Claimed mean
n = 25
x = 288000  # Sample mean
std = 80000  # Population standard deviation
a = .025  # Significance level (2.5%)
se = std / math.sqrt(n)  # Standard error
dist = stats.norm(mean, se)  # Normal distribution under null hypothesis
p = dist.cdf(x)  # Left-tailed p-value
print(p, p <= a)  # Accept or reject the null hypothesis

"""Hypothesis Test Right-tailed.

Problem: An expert claims that people who exercise regularly have a higher average oxygen intake than the general
         population, whose mean oxygen intake is 36.7 ml/kg. A sample of 15 exercisers shows an average intake of
         40.6 ml/kg with a sample standard deviation of 6 ml/kg. Test this claim at the 5% significance level.

Hypotheses:
    - H₀: The average oxygen intake of exercisers is equal to or less than the general population's average. μ <= 36.7
    - H₁: The average oxygen intake of exercisers is higher than the general population's average. μ > 36.7
"""
mean = 36.7  # Population mean
n = 15
x = 40.6  # Sample mean
s = 6  # Sample standard deviation
a = .05  # Significance level (5%)
se = s / math.sqrt(n)  # Standard error
dist = stats.t(n - 1, mean, se)  # T-distribution under null hypothesis
p = dist.sf(x)  # Right-tailed p-value
print(p, p <= a)  # Accept or reject the null hypothesis

"""Hypothesis Test for Proportion.

Problem: A judge claims that more than 25% of lawyers advertise their practice. A sample of 200 lawyers shows that
         63 advertise. Test the judge's claim at the 5% significance level.

Hypotheses:
    - H₀: The proportion of lawyers who advertise is less than or equal to 25%. p <= 25
    - H₁: The proportion of lawyers who advertise is greater than 25%. p > 25
"""
p = .25  # Claimed proportion
n = 200
px = 63 / n  # Sample proportion
a = .05  # Significance level (5%)
# Check if conditions for normal approximation are met
print(p * n >= 10 and (1 - p) * n >= 10)
se = math.sqrt((p * (1 - p)) / n)  # Standard error for proportion
dist = stats.norm(p, se)  # Normal distribution under null hypothesis
p_val = dist.sf(px)  # Right-tailed p-value
print(p_val, p_val <= a)  # Accept or reject the null hypothesis

"""Hypothesis Error Calculation.

Find out the hypothesis error from the following data:
- Population standard deviation (σ) = 100
- Sample size (n) = 100
- Significance level (α) = 0.05
- Null hypothesis mean (μ₀) = 30
- Alternative population mean (μ) = 26
"""
n = 100
mean = 30  # Null hypothesis mean
x = 26  # Alternative hypothesis mean
std = 100  # Population standard deviation
a = .05  # Significance level

se = std / math.sqrt(n)  # Standard error
z_critical = stats.norm.ppf(1 - a / 2)  # Z-critical value for two-tailed test
MoE = z_critical * se  # Margin of error
low = mean - MoE  # Lower bound of confidence interval
high = mean + MoE  # Upper bound of confidence interval
dist = stats.norm(x, se)  # Distribution under alternative hypothesis
beta = dist.cdf(high) - dist.cdf(low)  # Type II error probability (Beta)
print(se, z_critical, low, high, beta)  # Results for hypothesis error

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
prob = stats.poisson.pmf(k, mean_24)
print("Probability of getting 60 text in 24 hours:", prob)
