import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Upload data 
data = pd.read_csv("data/housing.csv") 

# Housing price distribution
plt.figure(figsize=(10, 6))
sns.histplot(data["median_house_value"], bins=50, kde=True)
plt.title("House price distribution")
plt.xlabel("Median house price")
plt.ylabel("Frequency")
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation matrix")
plt.show() 

# Relationship between median income and house prices
plt.figure(figsize=(10, 6))
sns.scatterplot(x="median_income", y="median_house_value", data=data)
plt.title("Median income vs. house prices") 
plt.xlabel("Median income")
plt.ylabel("Median house price")
plt.show() 

