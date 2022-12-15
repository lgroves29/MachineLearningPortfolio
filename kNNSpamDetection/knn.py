# Import block
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance
from sklearn.cluster import KMeans

# This is a file generated just for this lab 
import labseven

import random


def kNN(k, points):
    red = labseven.red_points(5)
    blue = labseven.blue_points(5)

    
