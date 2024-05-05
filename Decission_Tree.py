import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source

# Load the dataset
data = pd.read_csv('Genshin Impact Character Stats.csv')

# Define mappings for categorical data conversion to integers
element_mapping = {'Anemo': 1, 'Geo': 2, 'Electro': 3, 'Dendro': 4, 'Hydro': 5, 'Pyro': 6, 'Cryo': 7}
weapon_mapping = {'Sword': 1, 'Claymore': 2, 'Polearm': 3, 'Bow': 4, 'Catalyst': 5}
role_mapping = {'DPS': 1, 'Sub DPS': 2, 'Healer': 4, 'Support': 3}
ascension_mapping = {'HP': 1, 'ATK': 2, 'DEF': 3, 'Elemental Mastery': 4, 'Energy Recharge': 5, 
                     'CRIT Rate': 6, 'CRIT DMG': 7, 'Healing Bonus': 8, 'Anemo DMG': 9, 'Geo DMG': 10, 
                     'Electro DMG': 11, 'Dendro DMG': 12, 'Hydro DMG': 13, 'Pyro DMG': 14, 'Cryo DMG': 15, 'Physical DMG': 16}

# Apply mappings to the data
data['Element'] = data['Element'].map(element_mapping)
data['Weapon'] = data['Weapon'].map(weapon_mapping)
data['Main role'] = data['Main role'].map(role_mapping)
data['Ascension'] = data['Ascension'].map(ascension_mapping)

# Prepare features and target
inputColumnList = ['Rarity', 'Weapon', 'Main role', 'Ascension', 'Base HP', 'Base ATK', 'Base DEF']
outputColumnList = ['Element']  # Assuming 'Element' is the target
X = data[inputColumnList]
y = data['Element']

# Check for and handle NaNs by filling them with the median of each column
if X.isnull().any().any():
    X = X.fillna(X.median())
    print("NaN values filled with the median of their columns.")

# Handle missing values in the target if any
if y.isnull().any():
    y = y.fillna(y.mode()[0])
    print("NaN values in target filled with the mode.")

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X, y)

# Export the decision tree to a dot file and render it using graphviz
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=inputColumnList,
                           class_names=['Anemo', 'Geo', 'Electro', 'Dendro', 'Hydro', 'Pyro', 'Cryo'],
                           filled=True, rounded=True, special_characters=True)
graph = Source(dot_data)
graph.render("decision_tree", format='png', cleanup=True)  # This saves the output as 'decision_tree.png'

print("Decision tree model trained and visualization saved as 'decision_tree.png'")