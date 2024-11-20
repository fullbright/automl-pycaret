  
def split_sets(data_df, split_percent=0.8):
    # Spliting by training and test set
    # eighty_pct = 0.8*data_df.shape[0] 
    eighty_pct = split_percent * data_df.shape[0] 
    
    trainset_df = data_df.loc[:eighty_pct-1, :] 
    testset_df = data_df.loc[eighty_pct:, :] 
    return trainset_df, testset_df
