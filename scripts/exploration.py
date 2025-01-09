import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

# Charger les données
data = pd.read_csv("data/housing.csv")

# Distribution des prix des logements
plt.figure(figsize=(10, 6))
sns.histplot(data["median_house_value"], bins=50, kde=True)
plt.title("Distribution des prix des logements")
plt.xlabel("Prix médian des logements")
plt.ylabel("Fréquence")
plt.show()

# Matrice de corrélation
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Matrice de corrélation")
plt.show() 

# Relation entre le revenu médian et le prix des logements
plt.figure(figsize=(10, 6))
sns.scatterplot(x="median_income", y="median_house_value", data=data)
plt.title("Revenu médian vs Prix des logements")
plt.xlabel("Revenu médian")
plt.ylabel("Prix médian des logements")
plt.show()
