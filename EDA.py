import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Genshin Impact Character Stats.csv')

# Display the first few rows of the dataset
print(data.head())

# Summary statistics for numeric columns
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Data types of each column
print(data.dtypes)

# Histograms for numerical data
data.hist(bins=15, figsize=(15, 10))
plt.show()

# Plotting count plots for categorical data
for column in ['Element', 'Weapon', 'Main role', 'Ascension']:
    plt.figure(figsize=(10, 4))
    sns.countplot(x=column, data=data)
    plt.title(f'Distribution of {column}')
    plt.show()

    # Heatmap of correlations
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=2)
plt.show()

# Pair plots for selected numerical variables
sns.pairplot(data[['Base HP', 'Base ATK', 'Base DEF']])
plt.show()

# Box plots for numerical data by categories
plt.figure(figsize=(10, 6))
sns.boxplot(x='Element', y='Base ATK', data=data)
plt.title('Base ATK Distribution by Element')
plt.show()