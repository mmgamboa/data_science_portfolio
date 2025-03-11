import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def normalize_data(dataset):
    """
    This function receives a dataset and returns a new dataset with the columns processed.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset with the columns processed.        
    """
    preprocessing_steps = [
        proc_passengerId,
        proc_HomePlanet,
        proc_Destination,
        proc_CryoSleep,
        proc_Cabin
    ]
    
    for func in preprocessing_steps:
        dataset = func(dataset)
    
    return dataset

def proc_passengerId(dataset):
    """
    This function receives a dataset and returns a new dataset with the PassengerId column processed.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset with the PassengerId column processed.
    """
    # Check if the column exists
    if 'PassengerId' not in dataset.columns:
        print("PassengerId column not found.")
        return dataset
         
    ## PassengerId: Convert PassengerId by family 
    rename_passId = lambda name: int(name.split("_")[0])
    dataset['PassengerId']= dataset['PassengerId'].apply(rename_passId)
    
    return dataset

def proc_HomePlanet(dataset):
    """
    This function receives a dataset and returns a new dataset with the HomePlanet column processed.
    It generates new columns with the OneHotEncoding technique.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset with the HomePlanet column processed. New columns have prefix 'HomePlanet_'.
    """
    # Check if the column exists
    if 'HomePlanet' not in dataset.columns:
        print("HomePlanet column not found.")
        return dataset
    
    ## HomePlanet (keeping NaN)
    # Generate multiple new features according the HomePlannet (OneHotEncoding)
    mask_nan = dataset['HomePlanet'].isna()
    X_train_pp = pd.get_dummies(dataset, columns=['HomePlanet'], dtype=int)
    # Get all column names that start with 'HomePlanet_'
    homeplanet_columns = [col for col in dataset.columns if col.startswith('HomePlanet_')]
    dataset.loc[mask_nan, homeplanet_columns] = float('nan')

    return dataset

def proc_Destination(dataset):
    """
    This function receives a dataset and returns a new dataset with the Destination column processed.
    It generates new columns with the OneHotEncoding technique.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset with the Destination column processed. New columns have prefix 'Destination_'.
    """
    # Check if the column exists
    if 'Destination' not in dataset.columns:
        print("Destination column not found.")
        return dataset
    
    ## Destination (keeping NaN)
    # Generate multiple new features according the Destination (OneHotEncoding)
    mask_nan = dataset['Destination'].isna()
    X_train_pp = pd.get_dummies(dataset, columns=['Destination'], dtype=int)
    # Get all column names that start with 'HomePlanet_'
    destination_columns = [col for col in dataset.columns if col.startswith('Destination_')]
    dataset.loc[mask_nan, destination_columns] = float('nan')

    return dataset

def proc_CryoSleep(dataset):
    """
    This function receives a dataset and returns a new dataset with the CryoSleep column processed.
    It converts the column to an integer type.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset with the CryoSleep column processed.
    """
    # Check if the column exists
    if 'CryoSleep' not in dataset.columns:
        print("CryoSleep column not found.")
        return dataset
    
    ## CryoSleep & VIP
    #CryoSleep -> from bool to int
    label_mapping = {True: 1, False: 0}
    for ilabel in ['CryoSleep', 'VIP']:
        mask_nan = dataset[ilabel].isna()
        dataset[ilabel] = dataset[ilabel].map(label_mapping)
        dataset.loc[mask_nan, [ilabel]] = float('nan')
    
    return dataset

def proc_Cabin(dataset):
    """
    This function receives a dataset and returns a new dataset with the Cabin column processed.
    It generates new columns from Cabin to Deck, Num and Side.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset with the Cabin column processed. New columns are 'Deck', 'Num' and 'Side'.
    """
    # Check if the column exists
    if 'Cabin' not in dataset.columns:
        print("Cabin column not found.")
        return dataset
    
    ## Cabin
    # Split the Cabin column into three different columns = deck/num/side to have better resolution
    mask_nan = dataset['Cabin'].isna()
    dataset.Cabin = dataset.Cabin.replace(to_replace = np.nan, value = 'nan/nan/nan')
    dataset[['Deck', 'Num', 'Side']] = dataset['Cabin'].str.split('/', expand=True)
    for clabel in ['Deck', 'Num', 'Side']:
        dataset.loc[mask_nan, [clabel]] = float('nan')
    # Now let's use LabelEncoding for deck and side 
    label_encoder = LabelEncoder()
    # Fit the encoder and transform the 'deck' and 'side' columns
    dataset['Deck'] = label_encoder.fit_transform(dataset['Deck'])
    dataset['Side'] = label_encoder.fit_transform(dataset['Side'])
    dataset.drop(columns=['Cabin'], inplace=True)
    for clabel in ['Deck', 'Side', 'Num']:
        dataset.loc[mask_nan, [clabel]] = float('nan')
    
    return dataset

# Remove Name from X_train and X_test
X_train_pp = X_train.copy()
X_test_pp = X_test.copy()
X_train_pp.drop(columns='Name', inplace=True)
X_test_pp.drop(columns='Name', inplace=True)
#X_train_svc = normalize_data(X_train_pp)
X_train_ = normalize_data(X_train_pp)
X_test_ = normalize_data(X_test_pp)
print("Orig", train_data.isna().sum(axis=0).sum())
print("Prep", X_train_.isna().sum(axis=0).sum()+X_test_.isna().sum(axis=0).sum())
#print("Prep", X_test_.isna().sum(axis=0).sum())
print(201*2+199*2+182*2-200) #-200 because I deleted Name column for the analysis
print(3288-2324)
#