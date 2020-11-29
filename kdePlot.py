#Import libs
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
data = sns.load_dataset("mpg")

#print head
print(data.head())

#Generate the box/strip plot
sns.FacetGrid(data, hue="cylinders", size=6) \
                   .map(sns.kdeplot,"mpg") \
                   .add_legend()

#display plot
plt.show()