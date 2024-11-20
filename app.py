import os, ast
import pandas as pd
from utils import split_sets, train_model

dataset = os.getenv("INPUT_DATASET") # os.environ["INPUT_DATASET"]
target = os.getenv("INPUT_TARGET") # os.environ["INPUT_TARGET"]
usecase = os.getenv("INPUT_USECASE") # os.environ["INPUT_USECASE"]
github_branch = os.getenv("INPUT_GITHUB_BRANCH", "refs/heads/main")
split_percent = os.getenv("INPUT_DATASET_SPLIT_PERCENT", 0.8)

#print("Token INPUT_TOKEN: ", os.environ["INPUT_TOKEN"])
#print("Token GITHUB_TOKEN: ", os.environ["GITHUB_TOKEN"])
#print("Token INPUT_GITHUB_TOKEN: ", os.environ["INPUT_GITHUB_TOKEN"])

dataset_path = dataset # os.environ["INPUT_DATASET"] # + '.csv'
print("Dataset path 2 {}".format(dataset_path))

if usecase == 'regression':
    from pycaret.regression import *
elif usecase == 'classification':
    from pycaret.classification import *


train_model(dataset_path, usecase, target, split_percent)