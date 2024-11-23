import pandas as pd
import pycaret

def split_sets(data_df, split_percent=0.8):
    # Spliting by training and test set
    # eighty_pct = 0.8*data_df.shape[0] 
    eighty_pct = split_percent * data_df.shape[0] 
    
    trainset_df = data_df.loc[:eighty_pct-1, :] 
    testset_df = data_df.loc[eighty_pct:, :] 
    return trainset_df, testset_df


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

def update_dtypes(data_df):
    data_df['amount'] = data_df['amount'].apply(lambda x: x.replace(',','.'))
    data_df['amount'] = data_df['amount'].astype('float64')
    # data_df['Date'] =  data_df['Date'].astype('datetime64[ns]')
    data_df['Date'] =  pd.to_datetime(data_df['Date'], format="%d/%m/%Y") # df['TIME'] = pd.to_datetime(df['TIME'], format="%m/%d/%Y %I:%M:%S %p")
    data_df['name'] = data_df['name'].astype('string')
    data_df['merchant'] = data_df['merchant'].astype('string')
    data_df['Categorie'] = data_df['Categorie'].astype('string')
    data_df['Categorie2'] = data_df['Categorie2'].astype('string')
    data_df['Libellé'] = data_df['Libellé'].astype('float64')
    data_df['entree_sortie_argent'] = data_df['entree_sortie_argent'].astype('string')
    data_df['fixeOuVariable'] = data_df['fixeOuVariable'].astype('float64')
    data_df['FournisseurClient'] = data_df['FournisseurClient'].astype('string')
    data_df['TransactionId'] = data_df['TransactionId'].astype('string')
    data_df['category'] = data_df['category'].astype('string')
    data_df['BudgetMontant'] = data_df['BudgetMontant'].astype('float64')
    data_df['AnomalyAndComments'] = data_df['AnomalyAndComments'].astype('string')
    return data_df


def train_model(dataset_path, usecase, target, split_percent):

    model_final_name = "tuned_model_{}_{}".format(usecase, target)

    #data = pd.read_csv(dataset_path)
    # data = pd.read_csv("banking_data.csv")
    # data_df = pd.read_csv("banking_data_duplicated.csv")
    data_df = pd.read_csv(dataset_path)
    print(data_df.head())
    
    data_df = update_dtypes(data_df)

    print(data_df.dtypes)

    trainset_df, testset_df = split_sets(data_df, split_percent)
    print("Training set shape = {}, test set shape = {}".format(trainset_df.shape, testset_df.shape))

    if usecase == 'regression':
        # from pycaret.regression import *
        # exp1 = setup(data, target = target, session_id=123, silent=True, html=False, log_experiment=True, experiment_name='exp_github')
        # exp1 = setup(data, target = target, session_id=123, html=False, log_experiment=True, experiment_name='exp_github')
        exp1 = pycaret.regression.setup(trainset_df, target = target, session_id=123)

        # best = compare_models()
        best = pycaret.regression.compare_models(fold=3)

        # plot_model(best, plot="auc")

        # best_model = finalize_model(best)
        tuned = pycaret.regression.tune_model(best)

        # Predict unseen data
        predictions = pycaret.regression.predict_model(tuned, data=testset_df)

        pycaret.regression.save_model(tuned, model_final_name)

        logs_exp_github = pycaret.regression.get_logs(save=True)

    elif usecase == 'classification':
        # from pycaret.classification import *

        # exp1 = setup(data, target = target, session_id=123, silent=True, html=False, log_experiment=True, experiment_name='exp_github')
        # exp1 = setup(data, target = target, session_id=123, html=False, log_experiment=True, experiment_name='exp_github')
        exp1 = pycaret.classification.setup(trainset_df, target = target, session_id=123)
        top3 = pycaret.classification.compare_models(n_select = 3)
        tuned_top3 = [pycaret.classification.tune_model(i) for i in top3]
        blender = pycaret.classification.blend_models(tuned_top3)
        stacker = pycaret.classification.stack_models(tuned_top3)
        best_auc_model = pycaret.classification.automl(optimize = 'AUC')

        pycaret.classification.save_model(best_auc_model, "{}_best_auc".format(model_final_name))

        # best = compare_models()
        best = pycaret.classification.compare_models(fold=3)
        # plot_model(best, plot="auc")
        # best_model = finalize_model(best)
        tuned = pycaret.classification.tune_model(best)
        # Predict unseen data
        predictions = pycaret.classification.predict_model(tuned, data=testset_df)

        # save_model(best_model, 'model')
        pycaret.classification.save_model(tuned, model_final_name)


        # save_model(final_model,'Tuned Model 13 Nov 2021')

        logs_exp_github = pycaret.classification.get_logs(save=True)
