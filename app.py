import os, ast
import pandas as pd

dataset = os.environ["INPUT_DATASET"]
target = os.environ["INPUT_TARGET"]
usecase = os.environ["INPUT_USECASE"]
github_branch = os.getenv("INPUT_GITHUB_BRANCH", "refs/heads/main")
split_percent = os.getenv("INPUT_DATASET_SPLIT_PERCENT", 0.8)

#print("Token INPUT_TOKEN: ", os.environ["INPUT_TOKEN"])
#print("Token GITHUB_TOKEN: ", os.environ["GITHUB_TOKEN"])
#print("Token INPUT_GITHUB_TOKEN: ", os.environ["INPUT_GITHUB_TOKEN"])

def download_dataset():
    dataset_path = "https://raw.githubusercontent.com/" + os.environ["GITHUB_REPOSITORY"] + "/" + github_branch + "/" + os.environ["INPUT_DATASET"] + '?token=' + os.environ["INPUT_TOKEN"]
    print("Dataset path {}".format(dataset_path))
    print("Listing current files in the same directory")
    os.listdir()
    
    print("Listing recursively")
    files = []
    path = os.getcwd()
    print("Current working directory : {}".format(path))
    
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                files.append(os.path.join(r, file))
    
    # for f in files:
    #     print(f)
  

dataset_path = os.environ["INPUT_DATASET"] # + '.csv'
print("Dataset path 2 {}".format(dataset_path))

#data = pd.read_csv(dataset_path)
# data = pd.read_csv("banking_data.csv")
# data_df = pd.read_csv("banking_data_duplicated.csv")
data_df = pd.read_csv(dataset)
print(data_df.head())
print(data_df.dtypes)

# Spliting by training and test set
# eighty_pct = 0.8*data_df.shape[0] 
eighty_pct = split_percent * data_df.shape[0] 
  
trainset_df = data_df.loc[:eighty_pct-1, :] 
testset_df = data_df.loc[eighty_pct:, :] 
  
print("Training set shape = {}, test set shape = {}".format(trainset_df.shape, testset_df.shape))


if usecase == 'regression':
    from pycaret.regression import *
elif usecase == 'classification':
    from pycaret.classification import *

# exp1 = setup(data, target = target, session_id=123, silent=True, html=False, log_experiment=True, experiment_name='exp_github')
# exp1 = setup(data, target = target, session_id=123, html=False, log_experiment=True, experiment_name='exp_github')
exp1 = setup(trainset_df, target = target, session_id=123)

# best = compare_models()
best = compare_models(fold=3)

# plot_model(best, plot="auc")

# best_model = finalize_model(best)
tuned = tune_model(best)

# Predict unseen data
predictions = predict_model(tuned, data=testset_df)

# save_model(best_model, 'model')
save_model(tuned, 'tuned_model')
# save_model(final_model,'Tuned Model 13 Nov 2021')

logs_exp_github = get_logs(save=True)
