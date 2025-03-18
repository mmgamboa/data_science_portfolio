import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def encode_data(x_train_dataset, x_test_dataset):
    """
    This function receives two datasets and returns two new datasets with the columns processed.
    
    Parameters
    ----------
    x_train_dataset : pandas.DataFrame    
    x_test_dataset : pandas.DataFrame

    Returns
    -------
    x_train : pandas.DataFrame
        The dataset with the columns processed.
    x_test : pandas.DataFrame
        The dataset with the columns processed.
    """
    print("[Stage]: Encoding data ... ")    
    X_train = x_train_dataset.copy()
    X_test = x_test_dataset.copy()
    X_train.drop(columns='Name', inplace=True)
    X_test.drop(columns='Name', inplace=True)
    
    x_train, deck_col, side_col, labels_age_bins = encode(X_train, 
                                                        data_set='train')
    ## Store Transported column in y_train maintaining pandas.DataFrame format
    #y_train = pd.DataFrame(x_train['Transported'])
    ## Drop Transported column from x_train  
    #x_train.drop(columns=['Transported'], inplace=True)

    x_test = encode(X_test, 
                    data_set='test',
                    deck_col_train=deck_col,
                    side_col_train=side_col,
                    age_bins=labels_age_bins)
    x_test = x_test.reindex(columns=x_train.columns)
    
    print("[Stage]: Encoding data --> Done. ")    

    return x_train, x_test
    
def encode(dataset, 
           data_set='train',
           deck_col_train=None,
           side_col_train=None,
           age_bins=None,
           q=5):
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
    ]

    for func in preprocessing_steps:
        dataset = func(dataset)
    if data_set=='train':
        dataset, deck_col_train, side_col_train = proc_Cabin(dataset, 
                                                             data_set=data_set)
        dataset, labels_age_bins = proc_createFeatures(dataset, 
                                                       q=q, 
                                                       data_set=data_set)
        return dataset, deck_col_train, side_col_train, labels_age_bins
    else: 
        dataset = proc_Cabin(dataset, 
                             data_set,
                             deck_col_train=deck_col_train, 
                             side_col_train=side_col_train)
        dataset = proc_createFeatures(dataset, 
                                      q=q, 
                                      data_set=data_set, 
                                      label_age_bins=age_bins)
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
    else: 
        print(f"\t ... processing PassengerId column.")
                 
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
    else: 
        print(f"\t ... processing HomePlanet column.")
        
    ## HomePlanet (keeping NaN)
    # Generate multiple new features according the HomePlannet (OneHotEncoding)
    mask_nan = dataset['HomePlanet'].isna()
    dataset = pd.get_dummies(dataset, columns=['HomePlanet'], dtype=int)
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
    else: 
        print(f"\t ... processing Destination column.")
        
    ## Destination (keeping NaN)
    # Generate multiple new features according the Destination (OneHotEncoding)
    mask_nan = dataset['Destination'].isna()
    dataset = pd.get_dummies(dataset, columns=['Destination'], dtype=int)
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
    else: 
        print(f"\t ... processing CryoSleep column.")
    ## CryoSleep & VIP
    #CryoSleep -> from bool to int
    label_mapping = {True: 1, False: 0}
    for ilabel in ['CryoSleep', 'VIP']:
        mask_nan = dataset[ilabel].isna()
        dataset[ilabel] = dataset[ilabel].map(label_mapping)
        dataset.loc[mask_nan, [ilabel]] = float('nan')
    
    return dataset

def proc_Cabin(dataset, data_set='train', 
               deck_col_train=None, side_col_train=None):
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
    else: 
        print(f"\t ... processing Cabin column.")
        
    ## Cabin
    # Step 1: Split the Cabin column into Deck, Num, and Side
    dataset[['Deck', 'Num', 'Side']] = dataset['Cabin'].str.split('/', expand=True)
    
    # Step 2: Convert 'Num' to numeric (handling errors for NaN)
    dataset['Num'] = pd.to_numeric(dataset['Num'], errors='coerce')

    # Step 3: Handling NaN Values
    mask_nan = dataset['Cabin'].isna()
    dataset.loc[mask_nan, ['Deck', 'Num', 'Side']] = np.nan
    
    # Step 3: Handle missing values in 'Deck'
    # Create a boolean mask for rows where 'Deck' is NaN
	#mask_nan_deck = X['Deck'].isna()
	#mask_nan_deck = X['Side'].isna()
 
    # Step 4: One-hot encode 'Deck' and 'Side'
    dataset = pd.get_dummies(dataset, columns=['Deck'], drop_first=False)
    #print("Side uniquevalues", X['Side'].unique())
    dataset = pd.get_dummies(dataset, columns=['Side'], drop_first=False)

    # Get all new columns starting with Deck_
    deck_columns = [col for col in dataset.columns if col.startswith('Deck_')]
    # Get all new columns starting with Side_
    side_columns = [col for col in dataset.columns if col.startswith('Side_')]
    # Check that columns from test dataset are present in the train dataset
    if data_set == 'test':
        missing_deck = set(deck_col_train) -set(deck_columns)
        missing_side = set(side_col_train) -set(side_columns)
        print("Missing Deck", missing_deck)
        print("Missing Side", missing_side)
        for col in missing_deck:
            dataset[col] = np.nan
            deck_columns.append(col)
        for col in missing_side:
            dataset[col] = np.nan
            side_columns.append(col)
            
    # Step 5: Drop the original Cabin column
    dataset.drop(columns=['Cabin'], inplace=True)
    
    # Step 6: Ensure NaN values are preserved
    dataset.loc[mask_nan, ['Num']] = np.nan
    dataset[deck_columns] = dataset[deck_columns].astype(float)
    dataset[side_columns] = dataset[side_columns].astype(float)

    dataset.loc[mask_nan, deck_columns] = np.nan
    dataset.loc[mask_nan, side_columns] = np.nan
    if data_set == 'train':
        return dataset, deck_columns, side_columns
    else:
        return dataset

def proc_createFeatures(dataset,
                       verbose=False,
                       q=5,
                       data_set='train',
                       label_age_bins=None):
    """
    This function receives a dataset and returns a new dataset with the columns processed.
    It generates new columns from the columns 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
        
    Returns
    -------
    dataset : pandas.DataFrame
        The dataset with the new columns processed.
    """
    ## Create new features
    # Create a new feature that is the sum of all the services
    dataset = feature_TotalService(dataset)
    # Group the 'Age' feature into bins maintaining the proportions of number of passengers in each bin
    if data_set=='train':
        dataset, labels = feature_GroupByAge(dataset, q=q, data_set=data_set)
        return dataset, labels
    else:
        dataset = feature_GroupByAge(dataset, q=q, data_set=data_set, groupedage_columns_train=label_age_bins)
    
    return dataset

def feature_TotalService(dataset):
    # Check if the columns exist
    if 'RoomService' not in dataset.columns:
        print("RoomService column not found.")
        return dataset
    else: 
        print(f"\t ... processing RoomService column.")
        
    dataset['TotalServices'] = dataset['RoomService'] + dataset['FoodCourt'] + dataset['ShoppingMall'] + dataset['Spa'] + dataset['VRDeck']

    return dataset

def feature_GroupByAge(dataset, 
                q=5, 
                data_set='train',
                groupedage_columns_train=None):

	# Group the 'Age' feature into bins maintaining the proportions of number of passengers in each bin
	# Each bin has to have the same number of passengers
	if data_set=='train':
		# Step 1: Use pd.qcut with retbins=True
		dataset['GroupedAge'], bins = pd.qcut(dataset['Age'], q=q, retbins=True)

		# Step 2: Generate dynamic labels based on bin edges
		labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins)-1)]

		# Step 3: Reassign the binned data with dynamic labels
		dataset['GroupedAge'] = pd.qcut(dataset['Age'], q=q, labels=labels)

		# Step 4: One-hot encode 'GroupedAge' and 'Side'
		dataset = pd.get_dummies(dataset, columns=['GroupedAge'], drop_first=False)
		print("GropuedAge uniquevalues", labels)
		return dataset, labels
	# Get all new columns starting with GroupedAge_
	#groupedage_columns = [col for col in X.columns if col.startswith('GroupedAge_')]
	# Check that columns from test dataset are present in the train dataset
	if data_set == 'test':
		# Convert groupedage_columns_train into bins integers
		bins = [int(label.split('-')[0]) for label in groupedage_columns_train]
		bins.append(int(groupedage_columns_train[-1].split('-')[1]))
		dataset['GroupedAge'] = pd.cut(dataset['Age'], bins=bins, labels=groupedage_columns_train)
		
		dataset = pd.get_dummies(dataset, columns=['GroupedAge'], drop_first=False)

		return dataset
