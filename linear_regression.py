"""
1. Linear regression:
    Linear regression is a way to find a straight-line relationship between two variables.
    Equation:
           ŷ = mx + b
    Where: ŷ = Predicted value (dependent variable)
           x = Independent variable
           m = Slope of the regression line
           b = Intercept (value of Y when X = 0)

    Equation of slope:
           m = (y2 - y1) / (x2 - x1)
    Where: (x1, y1) and (x2, y2) are two points on the line.

2. Correlation Coefficient (r):
    The correlation coefficient (r) quantifies the strength and direction of the relationship between X and Y.
        - It always has a value between 1 and -1.
        - R > 0: Positive correlation (Y increases as X increases).
        - R < 0: Negative correlation (Y decreases as X increases).
        - R = 0: No correlation.

    The Pearson correlation coefficient equation:
           r = Σ((xᵢ - x̄) * (yᵢ - ȳ)) / (sqrt(Σ(xᵢ - x̄)²) * sqrt(Σ(yᵢ - ȳ)²))
    Where: xᵢ and yᵢ are individual data points in x and y, respectively
           x̄ is the mean of x
           ȳ is the mean of y

    The Pearson correlation coefficient equation using Z-score is:
           r = (1/n) Σ (Zₓ * Zᵧ)
    Where: Zₓ = (xᵢ - x̄) / s̄ₓ  (Z-score of xᵢ)
           Zᵧ = (yᵢ - ȳ) / s̄ᵧ  (Z-score of yᵢ)
           n is the number of data points

3. Residual:
    Residual is the difference between the actual observed value and the predicted value from the regression line:
        - eᵢ > 0: the model underestimates the actual value.
        - eᵢ < 0: the model overestimates the actual value.
    Equation:
           eᵢ = yᵢ - ŷᵢ
    Where: yᵢ = actual observed value
           ŷᵢ = predicted values

4. Sum of Squared Errors (SSE):
    The SSE is the sum of squared residuals.
        - It focuses on prediction error but doesn't explain how good the model is compared to a baseline.
        - The smaller the SSE, the better the model fits the data.
    Equation:
           SSE = Σ(yᵢ - ŷᵢ)²

5. Coefficient of Determination (r²):
    R-squared indicates how much variance in Y is explained by X. It measures how much prediction error was eliminated.
        - r² = 1 → Perfect fit (model explains 100% of the variance).
        - 0 < r² < 1 → Model explains some variance but not perfectly.
        - r² = 0 → Model explains nothing (as good as predicting the mean of y).
        - r² < 0 → Model is worse than predicting the mean of y (very poor fit).
    Equation:
           r² = r × r
    Where: r = Pearson correlation coefficient

    Equation:
           r² = 1 - (SSE / SST)
    Where: SSE (Sum of Squared Errors) = Σ(yᵢ - ŷᵢ)²
           SST (Total Sum of Squares) = Σ(yᵢ - ȳ)²

6. Mean Squared Error (MSE):
    Measures the average squared residuals values.
        - Lower MSE → Better model performance
        - Higher MSE → Poor fit (large deviations from actual values)
        - It is sensitive to outliers.
        -  MSE is frequently used as a loss function. During training, the model attempts to minimize the MSE by adjusting its parameters (e.g., weights and biases) to reduce the squared differences between predicted and actual values.
    Equation:
           MSE = (1/n) * Σ(yᵢ - ŷᵢ)²

7. Root Mean Squared Error (RMSE):
    Standard deviation of residuals - Square root of MSE, providing an error metric in the same units as Y:
        - The interpretation is easier than MSE.
    Equation:
           RMSE = sqrt(MSE) = sqrt((1/n) * Σ(yᵢ - ŷᵢ)²)

8. Standard error of the slope (stderr):
    It quantifies the precision of the estimated slope and indicates how much the estimated slope may differ from the true population slope if the sampling were repeated multiple times.
        - A smaller value of SE indicates a more reliable estimate of the slope.
        - The SE is used to construct confidence intervals and perform hypothesis tests about the slope.
    Equation:
           SE = s / sqrt(Σ(xᵢ - x̄)²)
    Where: s  = standard deviation of the residuals (errors)
           xᵢ = individual data points of the independent variable
           x̄  = mean of the independent variable

9. Standard error of the intercept (intercept_stderr):
    It quantifies the precision of the intercept estimate, indicating how much the estimated intercept may vary from the true population intercept if the sampling were repeated multiple times.
        - A smaller value of SEₛᵢ indicates a more reliable estimate of the intercept.
        - SEₛᵢ is used to construct confidence intervals for the intercept and perform hypothesis tests about the intercept.
    Equation:
        SEₛᵢ = s * sqrt( 1/n + (x̄² / Σ(xᵢ - x̄)²) )
    Where: s  = standard deviation of the residuals
           n  = number of data points
           x̄  = mean of the independent variable
           xᵢ = individual data points of the independent variable

10. Least Squares Method:
    The Least Squares Method determines the best-fitting line by minimizing the sum of squared differences between actual (yᵢ) and predicted (ŷᵢ) values. The goal is to reduce the total squared error.

    Best-fit slope (m) and intercept (b) formulas:
           m = (Σ(xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²
           b = ȳ - m * x̄
    Alternative formula using standard deviation and correlation coefficient:
           m = r * (sᵧ / sₓ)
    Where: r = Pearson correlation coefficient
           sₓ = Standard deviation of x-values
           sᵧ = Standard deviation of y-values
           x̄ = Mean of x-values
           ȳ = Mean of y-values

11.  Application:
    - Predictive Analytics → Forecasting sales, stock prices, and demand.
    - Econometrics → Understanding relationships between economic variables.
    - Engineering & Science → Analyzing experimental data.
    - Social Sciences → Studying human behavior patterns.
    - Machine Learning → Serving as a foundation for advanced models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 5, 7, 11])

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Calculate Coefficient of determination(r²)
r_squared = r_value ** 2

# Predict y values
y_pred = slope * x + intercept

# Calculate error
residuals = y - y_pred
mean_squared_error = np.mean(residuals ** 2)
root_mean_squared_error = np.sqrt(mean_squared_error)

# Plot using seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(x=x, y=y, color='blue', label='Data points')
sns.lineplot(x=x, y=y_pred, color='black', label='Regression line')

# Plot residuals as vertical and horizontal lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]],
             color='red', linestyle='--', linewidth=1)
    plt.plot([x[i], x[i]], [y_pred[i], y[i]],
             color='white', linestyle='--', linewidth=1)

# Annotate with regression equation, R-value, R-squared, and mean squared error
plt.text(2, 10, f'y = {slope:.2f}x {"+" if intercept >= 0 else "-"} {abs(intercept):.2f}',
         fontsize=12, color='black', ha='left')
plt.text(
    2, 9.5, f'r (Correlation coefficient): {r_value:.2f}', fontsize=12, color='black', ha='left')
plt.text(
    2, 9, f'r² (Coefficient of determination): {r_squared:.2f}', fontsize=12, color='black', ha='left')
plt.text(2, 8.5, f'Mean Squared Error: {mean_squared_error:.2f}',
         fontsize=12, color='black', ha='left')
plt.text(
    2, 8, f'Root Mean Squared Error: {root_mean_squared_error:.2f}', fontsize=12, color='black', ha='left')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.grid(False)  # Remove gridlines
plt.tight_layout()  # Ensures tight layout to prevent overlap
plt.show()
