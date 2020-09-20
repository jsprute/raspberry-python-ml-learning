import matplotlib.pyplot as plt
import seaborn as sns

print("Getting dataset")
iris = sns.load_dataset("iris")


#print("Print Head (first five records)")
#iris.head()

#scatter plot
#sns.jointplot(x="sepal_length", y="sepal_width", data=iris,size=6)

#facet grid plot
#sns.FacetGrid(iris, hue="species", size=6) \
#	.map(plt.scatter,"sepal_length","sepal_width") \
#	.add_legend()

#box plot
#sns.boxplot(x="species", y="sepal_length", data=iris)

# Strip Plot
#ax = sns.boxplot(x="species", y="sepal_length", data=iris)
#ax = sns.stripplot(x="species",y="sepal_length", data=iris, jitter=True, edgecolor="gray")

# Violin plot
#sns.violinplot(x="species", y="sepal_length", data=iris, size=6)

# KDE plot
#sns.FacetGrid(iris, hue="species", size=6) \
#	.map(sns.kdeplot, "sepal_length") \
#	.add_legend()

# Pair Plot
sns.pairplot(iris, hue="species", size=2.5)


plt.show()

