"""Basic Probability Rules

1. Probability of an event:
   P(A) = favorable_outcomes / total_outcomes

2. Complement Rule:
   P(Aᶜ) = 1 - P(A)

3. Addition Rule (Union of Two Events):
   P(A ∪ B) = P(A) + P(B) - P(A ∩ B)  # For any events.
   P(A ∪ B) = P(A) + P(B)  # For independent events.

4. Multiplication Rule:
     P(A ∩ B) = P(A) * P(B)  # For independent events.
     P(A ∩ B) = P(A) * P(B | A)  # For dependent events.
              = P(B) * P(A | B)

5. Conditional Probability:
     P(A | B) = P(A ∩ B) / P(B), given P(B) > 0
     P(A ∩ B) = P(B) * P(A | B)  # For dependent events.

6. Law of Total Probability:
   P(A) = Σ P(A | Bᵢ) * P(Bᵢ)

7. Bayes’ Theorem:
   P(Bᵢ | A) = (P(A | Bᵢ) * P(Bᵢ)) / Σ P(A | Bⱼ) * P(Bⱼ)
   P(B | A) = (P(A | B) * P(B)) / P(A)
   where P(B | A) is posterior probability,
         P(A | B) is likelihood
         P(A) is prior probability,
         P(B) is marginal probability.

Random Variables & Expectation:

8. Expected Value (Mean):
   - Discrete: E(X) = Σ x * P(X = x)
   - Continuous: E(X) = ∫ x * f(x) dx

9. Variance:
   Var(X) = E(X²) - [E(X)]²

10. Standard Deviation:
    σₓ = sqrt(Var(X))

11. Covariance:
    Cov(X, Y) = E[(X - E(X))(Y - E(Y))]

12. Correlation Coefficient:
    ρ(X,Y) = Cov(X, Y) / (σₓ * σᵧ)
"""

from math import comb
from math import perm

"""Probability of an Event

Problem: Given a fair die, what is the probability of rolling a 3?

Solution: Since there are 6 possible outcomes (1, 2, 3, 4, 5, 6) and only 1 favorable outcome (3), the probability is:
          P(rolling a 3) = 1/6.
"""
print("Probability of rolling a 3: ", 1/6)

"""Complement Rule

Problem: If the probability of it raining tomorrow is 0.4, what is the probability that it will not rain?

Solution: P(not raining) = 1 - P(raining) = 1 - 0.4 = 0.6.
"""
print("Probability of not raining: ", 1 - .4)

"""Addition Rule

Problem: A deck of cards has 52 cards. What is the probability of drawing a red card (hearts or diamonds) or a queen?

Solution: Let A be the event of drawing a red card, and B be the event of drawing a queen.
          P(A) = 26/52 (since half the deck is red cards)
          P(B) = 4/52 (since there are 4 queens in a deck)
          P(A ∩ B) = 2/52 (since there are 2 red queens)

          P(A ∪ B) = P(A) + P(B) - P(A ∩ B) = (26/52) + (4/52) - (2/52) = 28/52 = 7/13.
"""
print("Probability of drawing a red card or a queen: ",
      (26/52) + (4/52) - (2/52))

"""Multiplication Rule

Problem: A bag contains 3 red balls and 2 green balls. If you draw two balls without replacement,
         what is the probability that both are red?

Solution: The events are dependent, but applying the multiplication rule:
          P(first red) = 3/5
          P(second red | first red) = 2/4
          P(both red) = (3/5) * (2/4) = 6/20 = 3/10.
"""
print("Probability of both balls are red: ", (3/5) * (2/4))

"""Conditional Probability Calculation

Problem: A box contains 10 apples, 4 of which are rotten. The apples are randomly divided into two groups -
         Left and Right group that contains 5 apples (including 2 rotten and 3 fresh) on each side.
         If an apple is randomly selected from the right group, what is the probability that it is rotten?

Solution:
            | Group | Fresh | Rotten | Total |
            |-------|-------|--------|-------|
            | Right |   3   |    2   |   5   |
            | Left  |   3   |    2   |   5   |
            |-------|-------|--------|-------|
            | Total |   6   |    4   |  10   |


           P(rotten | right) = P(right ∩ rotten) / P(right) = (2/10) / (5/10) = 1/5 / 1/2 = 2/5 = 0.4.
"""
print("Probability of an apple from the right group is rotten: ", 2/5)

"""
6. Law of Total Probability

Problem: A factory has two machines: Machine 1 produces 70% of the total items and has a 95% success rate.
         Machine 2 produces 30% of the total items and has a 90% success rate.
         What is the total probability of producing a successful item?

Solution: Let A₁ be the event that an item is produced by Machine 1, and A₂ by Machine 2.
          P(success) = P(success | A₁) * P(A₁) + P(success | A₂) * P(A₂)
                     = (0.95 * 0.7) + (0.9 * 0.3) = 0.665 + 0.27 = 0.935.
"""
print("Total probability of producing a successful item: ",
      (0.95 * 0.7) + (0.9 * 0.3))

"""Bayes’ Theorem

Problem: In a city, 1% of residents have a rare disease, and 95% of people with the disease test positive for it,
         while 5% of people without the disease also test positive. What is the probability that a person who tests
         positive actually has the disease?

Solution: Let A be the event that the person has the disease, and B be the event that the person tests positive.
          P(A) = 0.01, P(Aᶜ) = .99, P(B | A) = 0.95, P(B | Aᶜ) = 0.05.
          P(A | B) = (P(B | A) * P(A)) / (P(B | A) * P(A) + P(B | Aᶜ) * P(Aᶜ))
                   = (0.95 * 0.01) / ((0.95 * 0.01) + (0.05 * 0.99)) ≈ 0.16.
"""
print("Probability of a person who tests positive actually has the disease: ",
      (0.95 * 0.01) / ((0.95 * 0.01) + (0.05 * 0.99)))

"""Expected Value (Mean) - Discrete Case

Problem: A discrete random variable X has the following probability distribution:
         | X  | 1   | 2   | 3   | 4   |
         |----|-----|-----|-----|-----|
         | P  | 0.1 | 0.3 | 0.4 | 0.2 |
         Compute the expected value E(X).

Solution: E(X) = Σ x * P(X = x) = (1×0.1) + (2×0.3) + (3×0.4) + (4×0.2) = 2.7
"""
X = [1, 2, 3, 4]
P = [0.1, 0.3, 0.4, 0.2]

print("Expected Value E(X): ", sum(x * p for x, p in zip(X, P, strict=True)))

"""Probability of an Event

Problem: When rolling a pair of six-sided dice, what is the probability of rolling a sum of 7?

Solution: Since there are 6x6 = 36 total possible outcomes when rolling two dices and favorable outcomes that
          sum to 7 are 6 {(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)}, the probability is:
          P(rolling a sum of 7) = 6/36 = 1/6.
"""
print("Probability of rolling a sum of 7: ", 6/36)

"""Combinatorial Probability

Problem: In a group of 10 people that have 6 adults and 4 children, if you were to choose 4 people randomly
         what is the probability there are an equal number of adults and children?

Solution: The total number of ways to choose 4 people from 10 is: C(10, 4) = 10! / (4! * (10 - 4)!) = 210
          The total number of ways to choose 2 adults from 6 adults is: C(6, 2) = 6! / (2! * (6 - 2)!) = 15
          The total number of ways to choose 2 children from 4 children is: C(4, 2) = 4! / (2! * (4 - 2)!) = 6
          Therefore, the probability is:
          P(2 adults and 2 children) = (15 * 6) / 210 = 3/7.
"""
print("The probability of selecting 2 adults and 2 children",
      (comb(6, 2) * comb(4, 2)) / comb(10, 4))

"""Combinatorial Probability

Problem: If 11 questions (easiest to hardest) are randomly ordered, what is the probability that
         the easiest and hardest question would be beside each other?

Solution: Since the easiest and hardest question must be beside each other, so they create a block of 2 questions.
          Within the block, the easiest and hardest questions can be arranged in: P(2, 2) = 2!
          10 questions can be arranged in: P(10, 10) = 10!
          11 questions can be arranged in: P(11, 11) = 11!
          Therefore, the probability is:
          P(easiest and hardest question beside each other) = (2! * 10!) / 11! = 2/11.
"""
print("The probability of easiest and hardest question beside each other",
      (perm(2, 2) * perm(10, 10)) / perm(11, 11))

"""Conditional Probability

Problem: Tina likes pasta. Tina's mother cooks pasta once a week. The probability that she cooks pasta and gives
         Tina ice cream as dessert is 2/3. On days when Tina's mother does not cook pasta, the probability that
         Tina will eat ice cream is 1/4. So
            1) What is the probability that Tina gets ice cream?
            2) What is the probability that she gets pasta and ice cream?

Solution: P(Pasta) = 1 / 7, P(No Pasta) = 1 - P(Pasta) = 1 - 1/7 = 6/7
          P(Ice cream | Pasta) = 2/3, P(Ice cream | No Pasta) = 1/4.
          1) P(Ice cream) = P(Pasta) * P(Ice cream | Pasta) + P(No Pasta) * P(Ice cream | No Pasta)
                        = 1/7 * 2/3 + 6/7 * 1/4 = 13/42.
          2) P(Pasta ∩ Ice cream) = P(Pasta) * P(Ice cream | Pasta)
                                   = 1/7 * 2/3 = 2/21.
"""
print("Probability of getting ice cream: ", 1/7 * 2/3 + (1 - 1/7) * 1/4)
print("Probability of getting pasta and ice cream: ", 1/7 * 2/3)
