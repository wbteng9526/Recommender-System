import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


HEADER = ['UserId', 'ItemId', 'rating', 'timestamp']
#path = "C:\\Users\\wbten\\OneDrive\\Desktop\\BU\\Fall 2018\\DS90-09\\GitHub_project"
path = "C:\\Users\\Wenbin Teng\\Desktop\\localfolder\\Python ML\\viax"

def load_data(ratio):
    print("Loading data...")
    
    rating_file = pd.read_csv(path + "\\data\\u.data", sep = '\t', names = HEADER)
    n_users = rating_file.UserId.unique().shape[0]
    n_items = rating_file.ItemId.unique().shape[0]
    
    train_data, test_data = train_test_split(rating_file, test_size = ratio)
    
    train_data_matrix = construct_matrix(train_data, n_users, n_items)
    test_data_matrix = construct_matrix(test_data, n_users, n_items)
    
    return train_data_matrix, test_data_matrix


def construct_matrix(df,nrow,ncol):
    df_matrix = np.zeros(shape = (nrow, ncol))
    for line in df.itertuples():
        df_matrix[line[1] - 1, line[2] - 1] = line[3]
        
    return df_matrix