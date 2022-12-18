import numpy as np
import pandas as pd
from helperFunctions import *



def test_data_wrangle_type():
    test_data = pd.DataFrame([["apple", 1], ["orange", 2]])
    test_data.columns = ["Fruit", "Ranking"]
    data_wrangle(test_data, ["Fruit"])
    assert isinstance(test_data.iloc[0,0], int)


def test_data_wrangle_shape():
    test_data = pd.DataFrame([["apple", 1], ["orange", 2]])
    test_data.columns = ["Fruit", "Ranking"]
    initial_shape = test_data.shape
    data_wrangle(test_data, ["Fruit"])
    assert initial_shape == test_data.shape

def test_class_mse():
    predictions = np.array([1,0,1,0])
    truth = np.array([0, 1,0,1])
    mse = classification_mse(predictions, truth)
    assert mse == 1

def test_cross_val():
    dog_full_pd = pd.read_csv("RandomForests/lab16data.csv", sep = ",", index_col = "Breed Name")
    dog_full_np = dog_full_pd.to_numpy(dtype = np.float16)
    in_dog_data = dog_full_np[:,:-1]
    out_class = dog_full_np[:,-1]
    cve = randomForestCV(in_dog_data, out_class, 5, 3)
    assert isinstance(cve, float)