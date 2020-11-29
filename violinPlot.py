#Import libs
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
data = sns.load_dataset("mpg")

#print head
print(data.head())

#Generate the box/strip plot
sns.violinplot(x="cylinders",y="mpg", data=data, size=6)

#display plot
plt.show()
