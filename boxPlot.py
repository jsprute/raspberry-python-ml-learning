#Import libs
import matplotlib.pyplot as plt
import seaborn as sns

#Load the data
data = sns.load_dataset("mpg")

#print head
print(data.head())

#Generate the box plot
sns.boxplot(x="cylinders",y="mpg", data=data)

#display plot
plt.show()




