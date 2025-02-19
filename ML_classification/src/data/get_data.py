import os

import pandas as pd

def get_paths():
    path_to_files = []
    for dirname, _, filenames in os.walk('data/'):
        for filename in filenames:
            path_to_files.append(os.path.join(dirname, filename))
    return path_to_files

def get_data(sub_train='train',
             sub_test='test'):
    
    path_to_files = get_paths()
    index_train = next(i for i, path in enumerate(path_to_files) if sub_train in path)
    index_test = next(i for i, path in enumerate(path_to_files) if sub_test in path)
    train_data = pd.read_csv(path_to_files[index_train])
    test_data = pd.read_csv(path_to_files[index_test])
    
    return train_data, test_data