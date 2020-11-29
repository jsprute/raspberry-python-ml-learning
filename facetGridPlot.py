#Import libs
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
data = sns.load_dataset("mpg")

#print head
print(data.head())

# Generate the Facet Grip Plot
sns.FacetGrid(data,hue="cylinders",size=6) \
    .map(plt.scatter, "weight","mpg") \
    .add_legend()

#Display the plot
plt.show()
    



