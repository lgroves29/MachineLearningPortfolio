## Code is adapted from this example in sklearn: 
#    https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
## Original adaptation by Kathleen R. Hablutzel
## Edits by Katherine M. Kinnaird

from matplotlib.colors import ListedColormap

# Create your list of labeled points 
# Note: you are deciding how many labeled points 
#       to have available

# Select your points 

# Create one array for those points
label_data = 

# Create the labels for the labeled points

# Hint: the above part should feel very familiar


# Run k-NN with the value of k that you think is best
# Note: the k here is the number of labeled points that 
#       you consult
# The output should be a list of labels for the 
#     unlabeled points called ALL_LABELS


all_labels = 


h = .02  # step size in the mesh

cmap_light = ListedColormap(['orange', 'cornflowerblue'])
cmap_bold = ['cyan', 'deeppink']

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = label_data[:, 0].min() - 1, label_data[:, 0].max() + 1
y_min, y_max = label_data[:, 1].min() - 1, label_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = NAME_OF_YOUR_KMEANS.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap=cmap_light)


# Plot the unlabeled points with their labels
sns.scatterplot(x=????[:, 0], y=????[:, 1], hue=????,
                palette="PiYG", alpha=1.0, edgecolor="black")

# Plot the labeled points with the output from your kNN
sns.scatterplot(x=????[:, 0], y=???[:, 1], hue=????,
                palette=cmap_bold, alpha=1.0, edgecolor="black")

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("2-Class classification (k = YOUR VALUE OF K)")