# Applying kNN to Spam Detection

This python notebook walks through the structure of the k-nearest-neighbors algorithm. kNN is _supervised classification_ algorithm, meaning that it trains on a set of data where entries include their classification, with the purpose of predicting the classification of new points. kNN bases its predictions of new points on their proximity to points whose classifications are known. The steps of the algorithm are as folllows:

1. Compute the distance from each point in the input set to the known points.
2. For each input point, find the k points who are closest in the known points.
3. Assign the input point to the class that has more of the k closest points (the k-nearest-neighbors)

Having built an understanding of how the kNN algorithm operates, the notebook goes on to apply it to a dataset consisting of emails with counts of how many times the most common words appear. These emails are classified as either _spam_ or regular emails. We can use this to construct a kNN classifier for new emails, which computes which other emails they are most similar to in word choice and classifies them accordingly.