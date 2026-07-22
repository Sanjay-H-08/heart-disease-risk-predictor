import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('heart.csv')

# 1. Target Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=data, palette='Set2')
plt.title('Class Balance (0 = Healthy, 1 = Heart Disease)')
plt.xlabel('Target')
plt.ylabel('Patient Count')
plt.show()

# 2. Annotated Correlation Heatmap
plt.figure(figsize=(11, 8))
corrmat = data.corr()
sns.heatmap(corrmat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation Matrix')
plt.show()

# 3. Boxplot: Max Heart Rate (thalach) by Target
plt.figure(figsize=(7, 5))
sns.boxplot(x='target', y='thalach', data=data, palette='Set2')
plt.title('Max Heart Rate Achieved (thalach) vs Heart Disease')
plt.xlabel('Target (0 = Healthy, 1 = Disease)')
plt.ylabel('Max Heart Rate')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# List of numerical features to inspect
num_features = ['age', 'chol', 'thalach']

for i, col in enumerate(num_features):
    sns.histplot(data[col], kde=True, ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()