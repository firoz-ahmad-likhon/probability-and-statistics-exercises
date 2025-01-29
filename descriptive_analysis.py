# Measure Usage Summary:
#
# | Measure               | Normal Data | Skewed Data  | Outliers Data |
# |-----------------------|-------------|--------------|---------------|
# | Mean                  | Yes         | No           | No            |
# | Median                | Yes         | Yes          | Yes           |
# | Mode                  | Yes         | Yes          | Yes           |
# | Variance              | Yes         | No           | No            |
# | Standard Deviation    | Yes         | No           | No            |
# | MAD (Mean)            | Yes         | No           | Yes           |
# | MAD (Median)          | Yes         | Yes          | Yes           |
# | IQR                   | No          | Yes          | Yes           |
# | Z-Score               | Yes         | No           | No            |

"""
Definitions:

Mean:
    The average value of a dataset.
    Formula:
        μ = (Σx) / n
    where x is the data points and n is the number of data points.

Median:
    The middle value of a dataset when arranged in ascending order.
    If the dataset has an even number of values, the median is the average of the two middle values.
    Formula:
        If n is odd: median = x[(n-1) // 2]
        If n is even: median = (x[(n//2)-1] + x[(n//2)]) / 2
    where x is the sorted data and n is the number of data points.

Mode:
    The most frequently occurring value(s) in a dataset.
    A dataset can have one mode, multiple modes, or no mode.

Variance:
    A measure of the spread of a dataset around the mean, calculated as the average of squared differences
    between each value and the mean.
    Formula:
        σ² = Σ(x - μ)² / n
    where x is the data points, μ is the mean, and n is the number of data points.

Standard Deviation:
    The square root of the variance, providing a measure of spread in the same units as the original data.
    Formula:
        σ = √(σ²)
    where σ² is the variance.

MAD (Mean):
    Mean Absolute Deviation, the average of the absolute differences between each data point and the mean,
    less sensitive to outliers than variance.
    Formula:
        MAD = (Σ|x - μ|) / n
    where x is the data points and μ is the mean.

MAD (Median):
    Median Absolute Deviation, the median of the absolute differences between each data point and the dataset's median,
    a robust measure of variability.
    Formula:
        MAD = median(|x - median(x)|)
    where x is the data points and median(x) is the median of the data.

Quartile:
    A quartile is a division of a dataset into four equal parts, each containing 25% of the observations.

IQR (Interquartile Range):
    The range between the first quartile (Q1, 25th percentile) and the third quartile (Q3, 75th percentile),
    used to measure the spread of the middle 50% of data and identify outliers.
    Formula:
        IQR = Q3 - Q1
    where Q1 is the first quartile and Q3 is the third quartile.

Z-Score:
    A standardized value representing how many standard deviations a data point is from the mean.
    Used for normal distributions to compare different datasets or detect outliers.
    Formula:
        z = (x - μ) / σ
    where x is the data point, μ is the mean, and σ is the standard deviation.

Percentile:
    A value below which a given percentage of data falls.
    For example, the 90th percentile is the value below which 90% of the data lies.
    Formula:
        Percentile(P) = (n + 1) * (P / 100)
    where P is the desired percentile (e.g., 25th, 50th, 75th), and n is the number of data points.

Ordinary Rank:
    A ranking method where tied values are assigned different ranks based on their order of appearance in the dataset.
    Tied values receive distinct ranks without skipping any numbers.
    Example: [10, 20, 20, 30] → Ranks: 1, 2, 3, 4.

Dense Rank:
    A ranking method where tied values share the same rank, and the next rank is assigned sequentially without skipping any numbers.
    Example: [10, 20, 20, 30] → Ranks: 1, 2, 2, 3.

Kurtosis:
    A measure of a distribution's tailedness and peak sharpness, indicating whether it has heavier or lighter tails and a sharper or flatter peak compared to a normal distribution.
    High kurtosis suggests more extreme outliers and a sharper peak, while low kurtosis suggests fewer outliers and a flatter peak.
    Formula:
        Kurtosis = (Σ(x - μ)⁴ / n) / (σ²)² - 3
    where x is the data points, μ is the mean, and σ² is the variance.

Skewness:
    A measure of the asymmetry of a distribution.
    Positive skewness indicates a longer or fatter tail on the right, while negative skewness indicates a longer or fatter tail on the left.
    Formula:
        Skewness = (Σ(x - μ)³ / n) / (σ³)
    where x is the data points, μ is the mean, and σ is the standard deviation.

Outlier Detection Equations:

1. IQR Method:
    - Lower Bound = Q1 - 1.5 * IQR
    - Upper Bound = Q3 + 1.5 * IQR
    - Outliers: Any data point < Lower Bound or > Upper Bound.

2. Z-Score Method:
    - z = (x - μ) / σ
    - Outliers: Any data point where |z| > 3 (for a 99.7% confidence interval).

3. Modified Z-Score Method:
    - modified_z = 0.6745 * (x - median) / MAD
    - Outliers: Any data point where |modified_z| > 3.5 (a typical threshold for identifying outliers).
"""

import numpy as np
from typing import Any
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns

speed: np.ndarray[Any, Any] = np.array(
    [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86])

print("-----Start Central Tendency-----")
mean = np.mean(speed)
median = np.median(speed)
mode = st.mode(speed)[0]
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("-----End Central Tendency-----\n\n")

print("-----Start Measure of the Spread-----")
_min = np.min(speed)
_max = np.max(speed)
_range = np.ptp(speed)
mid_range = (_min + _max) / 2
q25, q50, q75 = np.percentile(speed, [25, 50, 75], method='higher')
iqr = q75 - q25
variance = np.var(speed)  # Population variance
s_variance = np.var(speed, ddof=1)  # Sample variance
std = np.std(speed)  # Population standard deviation
s_std = np.std(speed, ddof=1)  # Sample standard deviation
mad_mean = np.mean(np.abs(speed - np.mean(speed)))
mad_median = np.median(np.abs(speed - np.median(speed)))
print('min', _min)
print('max', _max)
print('range', _range)
print('mid range', mid_range)
print('25%', np.percentile(speed, 25))
print('50%', q50)
print('75%', q75)
print('iqr', iqr)
print('iqr', st.iqr(speed))
print('p variance', variance)
print('s variance', s_variance)
print('p standard deviation', std)
print('s standard deviation', s_std)
print('mad(mean)', mad_mean)
print('mad(median)', mad_median)
print('mad(median) using function', st.median_abs_deviation(speed))
print("-----End Spread Analysis-----\n\n")

print("-----Start Data Modeling-----")
zscore = st.zscore(speed, ddof=1)
# zscore = (speed - np.mean(speed)) / np.std(speed, ddof=1)  # Using numpy
zscore = dict(zip(speed, zscore, strict=True))
zscore_86 = zscore[86]
loc = np.where(speed == 86)[0][0]
percentile_86 = st.percentileofscore(speed, 86)
ordinal_rank_86 = st.rankdata(speed, method='ordinal')[loc]
dense_rank_86 = st.rankdata(speed, method='dense')[loc]
print('percentile of 86 is', percentile_86)
print('ordinal ranking of 86 is', ordinal_rank_86)
print('dense ranking of 86 is', dense_rank_86)
print('z-score of 86', zscore_86)
print("-----End Data Modeling-----\n\n")

print("-----Start Shape-----")


def shape_interpretation(skewness: float, kurtosis: float) -> tuple[str, str]:
    """Interpret the distribution by skewness and kurtosis value.

    :param skewness: skewness value
    :param kurtosis: kurtosis value
    :return: String interpretation
    """
    # Interpret skewness
    if skewness == 0:
        skew_desc = "Symmetric distribution"
    elif 0 < skewness < 1:
        skew_desc = "Moderately right-skewed"
    elif skewness >= 1:
        skew_desc = "Highly right-skewed"
    elif -1 < skewness < 0:
        skew_desc = "Moderately left-skewed"
    else:  # skewness <= -1
        skew_desc = "Highly left-skewed"

    # Interpret kurtosis
    if kurtosis == 3:
        kurt_desc = "Mesokurtic (normal distribution)"
    elif kurtosis > 3:
        kurt_desc = "Leptokurtic (heavy tails)"
    else:  # kurtosis < 3
        kurt_desc = "Platykurtic (light tails)"

    return skew_desc, kurt_desc


skewness = st.skew(speed)
kurt = st.kurtosis(speed)
skew_desc, kurt_desc = shape_interpretation(skewness, kurt)
print(f"Skewness: {skewness:.2f}, {skew_desc}")
print(f"Kurtosis: {kurt:.2f}", kurt_desc)
print("-----End Shape-----\n\n")

print("-----Start Outlier Detection-----")


def remove_outliers_iqr(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The IQR method is a robust and effective tool for detecting and removing outliers. It is particularly useful in datasets with skewed distributions or when the data contains extreme values that do not reflect the overall pattern, helping to ensure more reliable and accurate analysis.

    Example:
    For a dataset of exam scores, there might be a few extreme outliers (e.g., a student who scored 1000 on a 100-point exam due to data entry error). Using the IQR method, we would identify and remove such values, ensuring that the remaining data represents the general performance of the students.

    :param data: numpy array of data
    :return: numpy array of removed outlier
    """
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    print('Removed outliers using IQR')
    return np.array([val for val in data if lower_bound <= val <= upper_bound])


def remove_outliers_zscore(data: np.ndarray[Any, Any], threshold: float = 3) -> np.ndarray[Any, Any]:
    """The Z-score is a useful method for detecting outliers in datasets that are approximately normally distributed. It quantifies how far a data point is from the mean in terms of standard deviations, making it easy to identify extreme values. However, it is most effective when the data is normally distributed, and may not be the best choice for skewed or non-normal datasets.

    Example:
    Consider a dataset of exam scores: [90, 92, 91, 87, 100, 500]. If we calculate the Z-scores for each value and find that 500 has a Z-score of 5, it is 5 standard deviations away from the mean. This suggests that 500 is an outlier, and we might choose to remove it to avoid skewing our analysis.

    :param data: numpy array of data
    :param threshold: the threshold for identifying outlier
    :return: numpy array of removed outlier
    """
    z_scores = np.abs(st.zscore(data, ddof=1))

    print('Removed outliers using Z-score')
    return np.array([val for key, val in enumerate(data) if z_scores[key] <= threshold])


def remove_outliers_modified_zscore(data: np.ndarray[Any, Any], threshold: float = 3.5) -> np.ndarray[Any, Any]:
    """The modified Z-score is more robust and reliable for detecting outliers, especially in skewed datasets or those with extreme values. It helps ensure that outlier detection is not disproportionately affected by extreme values.

    Example:
    If you have a dataset with some extreme values (outliers), calculating the Z-score might result in large values that incorrectly classify non-outliers as outliers. Using the modified Z-score instead will give a more accurate assessment of which values are truly outliers.

    :param data: numpy array of data
    :param threshold: the threshold for identifying outlier
    :return: numpy array of removed outlier
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad

    print('Removed outliers using modified Z-score')
    return np.array([val for key, val in enumerate(data) if np.abs(modified_z_scores[key]) <= threshold])


speed_iqr = remove_outliers_iqr(speed)
speed_zscore = remove_outliers_zscore(speed)
speed_modified_zscore = remove_outliers_modified_zscore(speed)
print("-----End Outlier Detection-----\n\n")

print("-----Start Visualizing-----")
sns.set_theme(style="whitegrid")  # Set up Seaborn style
# Create a figure with subplots
fig, axs = plt.subplots(3, 2, figsize=(16, 18))

# Plot 1: Annotations for Central Tendency
axs[0, 0].hist(speed, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
axs[0, 0].axvline(x=mean, color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {mean:.2f}')
axs[0, 0].axvline(x=median, color='green', linestyle='--',
                  linewidth=2, label=f'Median: {median}')
axs[0, 0].axvline(x=mode, color='purple', linestyle='--',
                  linewidth=2, label=f'Mode: {mode}')
axs[0, 0].set_title('Central Tendency Measures')
axs[0, 0].set_xlabel('Speed')
axs[0, 0].set_ylabel('Frequency')
axs[0, 0].legend()

# Plot 2: Annotations for Spread Analysis
axs[0, 1].boxplot(speed, vert=False, patch_artist=True,
                  boxprops=dict(facecolor='lightgreen'))
axs[0, 1].axvline(x=_min, color='blue', linestyle='--',
                  linewidth=2, label=f'Min: {_min}')
axs[0, 1].axvline(x=_max, color='orange', linestyle='--',
                  linewidth=2, label=f'Max: {_max}')
axs[0, 1].axvline(x=q25, color='magenta', linestyle='--',
                  linewidth=2, label=f'Q1: {q25}')
axs[0, 1].axvline(x=q50, color='brown', linestyle='--',
                  linewidth=2, label=f'Median: {q50}')
axs[0, 1].axvline(x=q75, color='cyan', linestyle='--',
                  linewidth=2, label=f'Q3: {q75}')
axs[0, 1].set_title('Spread Analysis Measures')
axs[0, 1].set_xlabel('Speed')
axs[0, 1].legend()

# Plot 3: Annotations for MAD and Variance
axs[1, 0].hist(speed, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
axs[1, 0].axvline(x=mean + mad_mean, color='red', linestyle='--', linewidth=2,
                  label=f'Mean + MAD(mean): {mean + mad_mean:.2f}')
axs[1, 0].axvline(x=mean - mad_mean, color='red', linestyle='--', linewidth=2,
                  label=f'Mean - MAD(mean): {mean - mad_mean:.2f}')
axs[1, 0].axvline(x=median + mad_median, color='green', linestyle='--', linewidth=2,
                  label=f'Median + MAD(median): {median + mad_median:.2f}')
axs[1, 0].axvline(x=median - mad_median, color='green', linestyle='--', linewidth=2,
                  label=f'Median - MAD(median): {median - mad_median:.2f}')
axs[1, 0].set_title('MAD Measures')
axs[1, 0].set_xlabel('Speed')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()

# Plot 4: Annotations for Data Modeling
axs[1, 1].hist(speed, bins=5, edgecolor='black', alpha=0.7, color='skyblue')
axs[1, 1].annotate(f'Percentile of 86: {percentile_86:.2f}%', xy=(86, 2), xytext=(100, 5),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].annotate(f'Ordinal Rank of 86: {ordinal_rank_86}', xy=(86, 3), xytext=(100, 6),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].annotate(f'Dense Rank of 86: {dense_rank_86}', xy=(86, 4), xytext=(100, 7),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].annotate(f'Z-score of 86: {zscore_86:.2f}', xy=(86, 5), xytext=(100, 8),
                   arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
axs[1, 1].set_title('Data Modeling Measures')
axs[1, 1].set_xlabel('Speed')
axs[1, 1].set_ylabel('Frequency')

# Plot 5: Bell Curve with Standard Deviation and Empirical Rules
sns.histplot(speed, kde=True, stat='density',
             linewidth=0, color='skyblue', ax=axs[2, 0])
xmin, xmax = axs[2, 0].get_xlim()
x = np.linspace(xmin, xmax, 100)
pdf = st.norm.pdf(x, mean, std)
axs[2, 0].plot(x, pdf, linewidth=2, color='red', label='Normal Distribution')
axs[2, 0].axvline(mean, color='k', linestyle='--', linewidth=1, label='Mean')
axs[2, 0].axvline(mean + std, color='blue', linestyle='--',
                  linewidth=1, label='Mean + Std')
axs[2, 0].axvline(mean - std, color='blue', linestyle='--',
                  linewidth=1, label='Mean - Std')
axs[2, 0].axvline(mean + 2 * std, color='green',
                  linestyle='--', linewidth=1, label='Mean + 2*Std')
axs[2, 0].axvline(mean - 2 * std, color='green',
                  linestyle='--', linewidth=1, label='Mean - 2*Std')
axs[2, 0].axvline(mean + 3 * std, color='purple',
                  linestyle='--', linewidth=1, label='Mean + 3*Std')
axs[2, 0].axvline(mean - 3 * std, color='purple',
                  linestyle='--', linewidth=1, label='Mean - 3*Std')
axs[2, 0].set_title('Bell Curve with Standard Deviation and Empirical Rules')
axs[2, 0].set_xlabel('Speed')
axs[2, 0].set_ylabel('Density')
axs[2, 0].legend()

# Plot 6: Bell Curve After Outlier Removal using Modified Z-score
sns.histplot(speed_modified_zscore, kde=True, stat='density',
             linewidth=0, color='skyblue', ax=axs[2, 1])
xmin, xmax = axs[2, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
pdf = st.norm.pdf(x, np.mean(speed_modified_zscore),
                  np.std(speed_modified_zscore, ddof=1))
axs[2, 1].plot(x, pdf, linewidth=2, color='red', label='Normal Distribution')
axs[2, 1].set_title('Bell Curve After Outlier Removal')
axs[2, 1].set_xlabel('Speed')
axs[2, 1].set_ylabel('Density')
axs[2, 1].legend()

plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between subplots
plt.show()
print("-----End Visualizing-----")
