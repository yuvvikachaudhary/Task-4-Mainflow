# Task-4-Mainflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a sample dataset
np.random.seed(42)

data = {
    'Age': np.random.normal(30, 5, 100),  # Normally distributed data
    'Income': np.random.normal(50000, 8000, 100),  # Normally distributed data
    'Expenses': np.random.normal(30000, 5000, 100),  # Normally distributed data
    'Scores': np.random.normal(75, 10, 100),  # Normally distributed data
    'Outliers': np.append(np.random.normal(50, 10, 95), [150, 200, 250, 300, 350])  # Adding outliers
}

df = pd.DataFrame(data)

# Histogram for Distribution
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.histplot(df['Age'], kde=True, color='blue', bins=15)
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
sns.histplot(df['Income'], kde=True, color='green', bins=15)
plt.title('Income Distribution')

plt.subplot(2, 2, 3)
sns.histplot(df['Expenses'], kde=True, color='red', bins=15)
plt.title('Expenses Distribution')

plt.subplot(2, 2, 4)
sns.histplot(df['Outliers'], kde=True, color='purple', bins=15)
plt.title('Outliers Distribution')

plt.tight_layout()
plt.show()

# Boxplot for Outliers
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.boxplot(x=df['Age'], color='blue')
plt.title('Age Boxplot')

plt.subplot(2, 2, 2)
sns.boxplot(x=df['Income'], color='green')
plt.title('Income Boxplot')

plt.subplot(2, 2, 3)
sns.boxplot(x=df['Expenses'], color='red')
plt.title('Expenses Boxplot')

plt.subplot(2, 2, 4)
sns.boxplot(x=df['Outliers'], color='purple')
plt.title('Outliers Boxplot')

plt.tight_layout()
plt.show()

# Pairplot for Relationships Between Variables
sns.pairplot(df[['Age', 'Income', 'Expenses', 'Scores', 'Outliers']], hue='Scores', diag_kind='kde')
plt.suptitle('Pairplot: Relationships Between Variables', y=1.02)
plt.show()

# Heatmap for Correlations
corr = df[['Age', 'Income', 'Expenses', 'Scores', 'Outliers']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
plt.title('Correlation Heatmap')
plt.show()
