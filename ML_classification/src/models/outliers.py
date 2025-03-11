import numpy as np
import pandas as pd

def outliers(dataset: pd.DataFrame,
             columns_outliers: list,
             detection_method:str,
             outlier_treatment:str,
             threshold_std: float=3,
             threshold_iqr:float=1.5,
             replace_by:str='mean',
             verbose:bool=False):
    """
    Detect and treat outliers in the dataset
    dataset: pd.DataFrame
    columns_outliers: list of str
    detection_method: str
        'std' for normal distribution
        'iqr' for skewed distribution
    outlier_threshold_std: float
        Threshold for normal distribution
    outlier_threshold_iqr: float
        Threshold for skewed distribution
    verbose: bool
        Print status of the process
        
    Returns the indexes of the outliers in the dataset for each column in columns_outliers
    """
    
    idx_train_data_outliers = outliers_inner(dataset, 
                                            columns_outliers, 
                                            detection_method,
                                            threshold_std=threshold_std,
                                            threshold_iqr=threshold_iqr,
                                            verbose=verbose)
    # Outliers treatment
    dataset_after_outliers = outliers_treatment(dataset,
                                               outlier_treatment,
                                               idx_train_data_outliers,
                                               replace_by=replace_by,
                                               verbose=verbose)
    return dataset_after_outliers
             

def outliers_inner(dataset: pd.DataFrame, 
            columns_outliers: list,
            method:str,
            threshold_std: float=3,
            threshold_iqr:float=1.5,
            verbose:bool=False):
    """
    Detect outliers in the dataset.
    dataset: pd.DataFrame
    columns_outliers: list of str
    method: str
        'std' for normal distribution
        'iqr' for skewed distribution
    threshold_std: float
        Threshold for normal distribution
    threshold_iqr: float
        Threshold for skewed distribution
    verbose: bool
        Print status of the process
    
    Returns indexes of the outliers in the dataset for each column in columns_outliers    
    """
    print("[Stage]: Outliers -> Detecting ")    
    
    total_length = dataset.shape[0]
    if method=='std':
        print("... with Std method")
        idx_outliers = outliers_norm_distro(dataset, 
                                            columns_outliers,
                                            threshold=threshold_std,
                                            verbose=verbose)
    
    elif method=='iqr':
        print("... with IQR method")
        idx_outliers = outliers_skewed_distro(dataset, 
                                              columns_outliers,
                                              threshold=threshold_iqr,
                                              verbose=verbose)
        
    print("[Stage]: Outliers -> Done.")
    
    return idx_outliers 

def outliers_norm_threshold(dataset, 
                            columns,
                            threshold=3,
                            verbose=False):
    """
    Compute the threshold for normal distribution for each column in columns
    dataset: pd.DataFrame
    columns: list of str
    threshold: float
    
    Returns low_value, high_value
    """
    print(f"\tComputing threshold for normal distribution...")
    aux_data = dataset.copy()
    aux_data = aux_data[columns]

    mean_ds = aux_data.mean(axis=0)
    std_ds = aux_data.std(axis=0)
    low_value = {}
    high_value = {}
    for j, name in enumerate(mean_ds.keys()):
        low_value[name] = max(mean_ds[name] - threshold * std_ds[name], 0)
        high_value[name] = mean_ds[name] + threshold * std_ds[name]
        if verbose: 
            print(f"\t\t{name}: {low_value[name]}, {high_value[name]}")
            
    return low_value, high_value

def outliers_norm_distro(dataset, 
                         colnames,
                         threshold=3,
                         verbose=False):
    """
    Detect outliers in the dataset.
    
    dataset: pd.DataFrame
    cols: pd.Series or list of column names where to search outliers
    -
    Returns indexes where the condition is not satisficed in each col in cols
    """
    low_norm, high_norm = outliers_norm_threshold(dataset, 
                                                  colnames,
                                                  threshold=threshold,
                                                  verbose=verbose)

    aux_data_norm={}
    for j, name in enumerate(low_norm.keys()):
        aux_data_norm[name] = dataset[(dataset[name] < low_norm[name]) | (dataset[name] > high_norm[name]) ].index
    return aux_data_norm

# IQR
def outliers_skewed_threshold(dataset, 
                              colnames,
                              threshold=1.5,
                              verbose=False):
    """
    Returns the low_limit and up_limit for each column in colnames
    
    dataset: pd.DataFrame
    colnames: list of str
    threshold: float
    
    Returns low_limit, up_limit    
    """
    aux_data_q = dataset.copy()
    quantile1 = aux_data_q[colnames].quantile(0.25)
    quantile3 = aux_data_q[colnames].quantile(0.75) 
    #print(quantile1, quantile3)
    # compute iqr
    iqr = quantile3 - quantile1
    up_limit_q = quantile3 + threshold * iqr
    low_limit_q = quantile1 - threshold * iqr
    for j, name in enumerate(low_limit_q.keys()):
        low_limit_q[name] = max(low_limit_q[name], 0)
    return low_limit_q, up_limit_q

def outliers_skewed_distro(dataset, 
                           colnames,
                           threshold=1.5):
    """
    Detect outliers in the dataset.
    
    dataset: pd.DataFrame
    cols: pd.Series or list of column names where to search outliers
    -
    Returns indexes where the condition is not satisficed in each col in cols
    """    
    low_limit, up_limit = outliers_skewed_threshold(dataset, 
                                                    colnames,
                                                    threshold=threshold)
    aux_data_norm={}
    for j, name in enumerate(low_limit.keys()):
        aux_data_norm[name] = dataset[(dataset[name] < low_limit[name]) | (dataset[name] > up_limit[name]) ].index
    return aux_data_norm

#==============================================================================
# Outliers Treatment
#==============================================================================
def outliers_treatment(dataset,
                       method,
                        idx_outliers,
                        replace_by='mean',
                        verbose = False):
    """
    Treat the outliers in the dataset
    dataset: pd.DataFrame
    method: str
        'nan' for setting the outliers as NaN
        'remove' for removing the outliers
        'replace' for replacing the outliers with the mean or median
    idx_outliers: dict
        Dictionary with the indexes of the outliers for each column in the dataset
    verbose: bool
    
    Returns the dataset with the outliers treated
    """
    print("[Stage]: Outliers -> treating")
    if verbose: print(f"\t Original total length {dataset.shape[0]}")
    
    if method == 'nan':
        if verbose: print("... setting outliers as NaN")
        dataset = set_outliers_as_nan(dataset, 
                                      idx_outliers, 
                                      verbose=verbose)
    elif method == 'remove':
        if verbose: print("... removing outliers")
        dataset = set_outliers_as_remove(dataset, 
                                         idx_outliers, 
                                         verbose=verbose)
    elif method == 'replace':
        if verbose: print("... replacing outliers")
        dataset = set_outliers_as_replace(dataset, 
                                          idx_outliers, 
                                          method_to_replace=replace_by,
                                          verbose=verbose)
    else:
        raise ValueError("Method must be 'nan', 'remove' or 'replace'")

    if verbose: print(f"\t New total length {dataset.shape[0]}")
    
    print("[Stage]: Outliers -> Done.")
    return dataset

def set_outliers_as_nan(dataset, idx_outliers, verbose=False):
    """
    Approach: 
        Assign NaN to the outliers in the dataset
    dataset: pd.DataFrame
    idx_outliers: dict
        Dictionary with the indexes of the outliers for each column in the dataset
        
    Returns the dataset with the outliers set as NaN
    """
    for name in idx_outliers.keys():
        dataset.loc[idx_outliers[name], name] = np.nan
    return dataset

def set_outliers_as_remove(dataset, idx_outliers, verbose=False):
    """
    Approach:
        Remove the outliers from the dataset
    dataset: pd.DataFrame
    idx_outliers: dict
        Dictionary with the indexes of the outliers for each column in the dataset
        
    Returns the dataset without the outliers
    """
    
    # There are some indexes that are repeated in the dictionary idx_outliers. So we want to remove only the unique indexes
    # because inplace remove the indexes from the dataset, when we remove the first index, the second index will be different
    # from the original index. So we need to remove only the unique indexes
    idx_to_remove = []
    for name in idx_outliers.keys():
        idx_to_remove.extend(idx_outliers[name])
    idx_to_remove = list(set(idx_to_remove))
    
    total_length = dataset.shape[0]
    total_removed = len(idx_to_remove)
    dataset.drop(idx_to_remove, inplace=True)
    if verbose:
        print(f"\t ~{total_removed/total_length*100:3.0f} % of dataset removed")
    
    return dataset 

def set_outliers_as_replace(dataset, 
                            idx_outliers, 
                            method_to_replace='mean',
                            verbose=False):
    """
    Approach:
        Replace the outliers with the mean or median of the column
    dataset: pd.DataFrame
    idx_outliers: dict
        Dictionary with the indexes of the outliers for each column in the dataset
    method: str
        'mean' or 'median'

    Returns the dataset with the outliers replaced
    """
    for name in idx_outliers.keys():
        if method_to_replace == 'mean':
            replacement_value = dataset[name].mean()
        elif method_to_replace == 'median':
            replacement_value = dataset[name].median()
        else:
            raise ValueError("Method must be 'mean' or 'median'")
        
        dataset.loc[idx_outliers[name], name] = replacement_value
    return dataset