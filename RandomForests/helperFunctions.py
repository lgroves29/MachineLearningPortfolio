# easy way to make categories numeric
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier

def data_wrangle(dataset, lst):

    for column in lst:
		# get list of possible values
        unique_val = dataset[column].unique()
        n_unique = len(unique_val)

		# assign each unique value a unique numerical value
        replacements = dict(zip(unique_val, range(n_unique)))

		# replace each string value with its corresponding numerical value
        for var in replacements.keys():
            dataset[column][dataset[column] == var] = replacements[var]

    return dataset

# compute error for classification
def classification_mse(class_truth, pred_class):
    indicators = np.zeros(len(class_truth))
    index = class_truth == pred_class
    indicators[index] = 0
    indicators[~index] = 1
    return np.mean(indicators)

# compare models with different hyperparameters:
def randomForestCV(inputs, output, k, n_estimators = 10, max_features = 10, max_depth = 10):
    """
    args:
    inputs - columns of a numpy array to train model on
    outputs - class variable
    k - number of sections to split code into
    """


    #divide data into k sets
    # want set size to be the smallest int n st kn > len(data)
    set_size = math.ceil(len(inputs)/k )

	# repeat for each of the k test sets
    test_errors = []

    for i in range(k):
		# grab the ith chunk to be the test data 
        test_in = inputs[i*set_size:set_size+i*set_size, :]
        test_out = output[i*set_size:set_size+i*set_size]

		# everything else is training data
        train_in = np.vstack((inputs[:i*set_size,:], inputs[set_size+i*set_size:, :]))
        train_out = np.concatenate((output[:i*set_size], output[set_size+i*set_size:]))
		# print(train.shape)
		
		# Fit model to training dataset
        grove = RandomForestClassifier(n_estimators= n_estimators, max_features= max_features, max_depth= max_depth, random_state=0)
        grove.fit(train_in, train_out)

		# Compute the testing error
        test_preds = grove.predict(test_in)
        test_error = classification_mse(test_preds, test_out)
        test_errors.append(test_error)

    cross_val = np.mean(test_errors)
    return cross_val


# Helper functions
# Our first implementation will be built from scratch
# functions from Lab 16

def gini_index(data_pd: pd.DataFrame, class_var: str) -> float:
    """
    Given the observations of a binary class and the name of the binary class column
    calculate the gini index
    """
    # count classes 0 and 1
    count_A = np.sum(data_pd[class_var] == 0)
    count_B = np.sum(data_pd[class_var])

    # get the total observations
    n = count_A + count_B

    # If n is 0 then we return the lowest possible gini impurity
    if n == 0:
        return 0.0

    # Getting the probability to see each of the classes
    p1 = count_A / n
    p2 = count_B / n

    # Calculating gini
    gini = 1 - (p1 ** 2 + p2 ** 2)

    # Returning the gini impurity
    return gini

def info_gain(data_pd: pd.DataFrame, class_var: str, feature: str) -> float:
    """
    Calculates how much info we gain from a split compared to info at the current node
    """
    # compute the base gini impurity (at the current node)
    gini_base = gini_index(data_pd, class_var)

    # split on the feature
    node_left, node_right = split_bool(data_pd, feature)

    # count datapoints in each split and the whole dataset
    n_left = node_left.shape[0]
    n_right = node_left.shape[0]
    n = n_left + n_right

    # get left and right gini index
    gini_left = gini_index(node_left, class_var)
    gini_right = gini_index(node_right, class_var)

    # calculate weight for each node
    # according to proportion of data it contains from parent node
    w_left = n_left / n
    w_right = n_right / n

    # calculated weighted gini index
    w_gini = w_left * gini_left + w_right * gini_right

    # calculate the gain of this split
    gini_gain = gini_base - w_gini

    # return the best feature
    return gini_gain

def split_bool(data_pd, column_name):
    """Returns two pandas dataframes:
    one where the specified variable is true,
    and the other where the specified variable is false"""
    node_left = data_pd[data_pd[column_name]]
    node_right = data_pd[~data_pd[column_name]]
    
    return node_left, node_right

def best_split(data_pd: pd.DataFrame, class_var: str, exclude_features: list = []) -> float:
    """
    Returns the name of the best feature to split on at this node.
    If the current node contains the most info (all splits lose information), return None.
    EXCLUDE_FEATURES is the list of variables we want to omit from our list of choices
    """
    # compute the base gini index (at the current node)
    gini_base = gini_index(data_pd, class_var)

    # initialize max_gain and best_feature
    max_gain = 0
    best_feature = None

    # create list of features of data_pd not including class_var
    # print(data_pd.columns)
    # print(class_var)
    # features = list(set(data_pd.columns).difference(set([class_var])))
    features = [f for f in np.array(data_pd.columns) if f not in np.array(class_var)]
    
    
    # This line will be useful later - can skip for now
    # remove features we're excluding
    # (already made decision on this feature)
    features = [f for f in features if f not in exclude_features]

    # test a split on each feature
    # info_gain(dog_pd, 'Easy To Train', 'Good For Novice Owners')
    for feature in features:
        info = info_gain(data_pd, feature, class_var)

        # check whether this is the greatest gain we've seen so far
        # and thus the best split we've seen so far
        if info > max_gain:
            best_feature = feature
            max_gain = info

    # return the best feature
    return best_feature

def build_decision_tree(node_data: pd.DataFrame, class_var: str, depth: int = 0, exclude_features: list = []) -> None:
    """Build a decision tree for NODE_DATA with 
    CLASS_VAR as the variable that stores the class assignments. 
    EXCLUDE_FEATURES is the list of variables we want to omit from our list of choices"""
    # 0. stop at the base case
    max_depth = 2
    if depth >= max_depth:
        return
    
    # 1. determine which decision gives us the most information
    best_feature = best_split(node_data, class_var, exclude_features)
    print(f"{'>'*(depth+1)}Splitting {node_data.shape[0]} data points on {best_feature}")
    
    # 2a. if best_feature == None, don't split further
    if best_feature == None:
        print(f"{'>'*(depth+1)}No best next split.")
        return
    
    # 2b. else, make the split according to the best decision
    else:
        data_left, data_right = split_bool(node_data, best_feature)
        print(f"{'>'*(depth+1)}Produces {data_left.shape[0]} True data points and {data_right.shape[0]} False data points")
        
        # and exclude this feature at future levels of the tree
        exclude_features.append(best_feature)
    
    # 3. continue recursively on each of the resulting two nodes
    build_decision_tree(data_left, class_var, depth + 1, exclude_features)
    build_decision_tree(data_right, class_var, depth + 1, exclude_features)
    return