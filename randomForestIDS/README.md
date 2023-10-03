# Random Forests for Intrusion Detection

An *intrusion detection system* is a system that monitors network or computer traffic for malicous behavior. In this notebook I explain how machine learning can be applied to predict which network behavior is malicious or normal. To do this I will use random forests. Random Forests are an ensemble machine learning method that aggregates the predictions of multiple decision trees. First I will go over how random forests are structured with a simple dataset example. Then I will apply this idea to a dataset of network traffic logs, which are classificed as malicious or benign. Training a random forest classifier on this dataset will allow me to create a basic intrusion detection system.