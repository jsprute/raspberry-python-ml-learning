# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Load the CSV dataset
data = pd.read_csv('./data/diabetes.csv')
print(data.head(10))
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Impute the missing vlaues using the feature median values
imputer = SimpleImputer(missing_values=0, strategy="median")
X_train2 = imputer.fit_transform(X_train)
X_test2 = imputer.transform(X_test)

# Convert the numpy array into a Dataframe
X_train3 = pd.DataFrame(X_train2)

# Display the first 10 records
print(X_train3.head(10))


# Define a histogram plot method
def plotHistogram(values, label, feature, title):
    sns.set_style("whitegrid")
    plotOne = sns.FacetGrid(values, hue=label, aspect=2)
    plotOne.map(sns.distplot, feature, kde=False)
    plotOne.set(xlim=(0, values[feature].max()))
    plotOne.add_legent()
    plotOne.set_axis_labels(feature, 'Proportion')
    plotOne.fig.suptitle(title)
    plt.show()

# Plot the Insulin histogram
plotHistogram(X_train3, None, 4, 'Insulin vs Diagnosis')

# Plot the SkinThickness histogram
plotHistogram(X_train3, None, 3,'SkinThickness vs Diagnosis')

# Summary of the number of 0's present in the dataset by feature
data2 = X_train2
print("Num of Rows, Num of Columns: ", data2.shape)
print("\nColumn Name            Num of Null Values\n")
print((data2[:] == 0).sum())

# Percentage summary of the number of 0's in the dataset
print("Num of Rows, Num of Columns: ", data2.shape)
print("\nColumn Name          %Null Values\n")
print(((data2[:] == 0).sum())/ 614 * 100)

# Create a heat map
#g = sns.heatmap(dataset.corr(), cmap="BrBG", annot=False)
#plt.show()

# Display the feature correlation values
corr1 = X_train3.corr()
print(corr1[:])

