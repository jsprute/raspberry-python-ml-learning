#import required libraries
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
data = sns.load_dataset("mpg")

#print head
print(data.head())

#Generate the scatter plot
sns.jointplot(x="weight",y="mpg", data=data,size=6)

#Display the plots
plt.show()

