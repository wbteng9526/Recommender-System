import numpy as np
import pandas as pd


HEADER = ['UserId', 'ItemId', 'rating', 'timestamp']
#path = "C:\\Users\\wbten\\OneDrive\\Desktop\\BU\\Fall 2018\\DS90-09\\GitHub_project"
path = "C:\\Users\\Wenbin Teng\\Desktop\\localfolder\\Python ML\\viax"


def load_data(args):
    print('loading data ...')

    rating_file = path + "\\data\\u.data"
    rating_np = pd.read_csv(rating_file,sep = '\t', names = HEADER)
    rating_np = rating_np.values
    rating_np = rating_np[:, 0:3]  # remove timestamp
    rating_np[:, 0:2] = rating_np[:, 0:2] - 1
    n_user = np.max(rating_np[:, 0]) + 1
    n_item = np.max(rating_np[:, 1]) + 1
    train, val, test = dataset_split(rating_np,args)
    return int(n_user), int(n_item), train, val, test


def dataset_split(rating_np,args):
    print('splitting dataset ...')

    # train : validation : test = 6 : 2 : 2
    n_ratings = rating_np.shape[0]

    val_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * args.val_ratio), replace=False)
    left = set(range(n_ratings)) - set(val_indices)
    test_indices = np.random.choice(list(left), size=int(n_ratings * args.test_ratio), replace=False)
    train_indices = list(left - set(test_indices))

    train = rating_np[train_indices]
    val = rating_np[val_indices]
    test = rating_np[test_indices]

    return train, val, test
