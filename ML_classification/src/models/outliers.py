import pandas as pd

def outliers(dataset: pd.DataFrame, 
            columns_outliers: list,
            method:str,
            threshold: float=3,
             verbose=False):
    """
    Detect outliers in the dataset.
    """
    
    total_length = dataset.shape[0]
    if method=='std':
        idx_outliers = outliers_norm_distro(dataset, 
                                                 columns_outliers,
                                                 threshold=threshold,
                                                 verbose=verbose)
    
        #len_norm = {ikey: len(ival) for ikey, ival in zip(idx_outliers.keys(), 
        #                                          idx_outliers.values())}
        #counts_norm = list(len_norm.values())

    elif method=='iqr':
        idx_outliers = outliers_skewed_distro(dataset, columns_outliers)
        
        #len_skewed = {ikey: len(ival) for ikey, ival in zip(idx_outliers.keys(), 
        #                                      idx_outliers.values())}
        #counts_skewed = list(len_skewed.values())
        
    return idx_outliers 

def outliers_norm_threshold(dataset, 
                            columns,
                            threshold=3):
    """
    
    """
    aux_data = dataset.copy()
    aux_data = aux_data[columns]
    #print(aux_data)
    mean_ds = aux_data.mean(axis=0)
    std_ds = aux_data.std(axis=0)
    #mean_ds, std_ds
    low_value = {}
    high_value = {}
    for j, name in enumerate(mean_ds.keys()):
        low_value[name] = max(mean_ds[name] - threshold * std_ds[name], 0)
        high_value[name] = mean_ds[name] + threshold * std_ds[name]
        
    return low_value, high_value

def outliers_norm_distro(dataset, 
                         colnames,
                         threshold=3,
                         verbose=False):
    """ 
    dataset: pd.DataFrame
    cols: pd.Series or list of column names where to search outliers
    -
    Returns indexes where the condition is not satisficed in each col in cols
    """
    low_norm, high_norm = outliers_norm_threshold(dataset, 
                                                  colnames,
                                                  threshold=threshold)
    #low_norm , high_norm
    aux_data_norm={}
    for j, name in enumerate(low_norm.keys()):
        aux_data_norm[name] = dataset[(dataset[name] < low_norm[name]) | (dataset[name] > high_norm[name]) ].index
    return aux_data_norm

# IQR
def outliers_skewed_threshold(dataset, 
                              colnames,
                              threshold=1.5):
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
    
    low_limit, up_limit = outliers_skewed_threshold(dataset, 
                                                    colnames,
                                                    threshold=threshold)
    aux_data_norm={}
    for j, name in enumerate(low_limit.keys()):
        aux_data_norm[name] = dataset[(dataset[name] < low_limit[name]) | (dataset[name] > up_limit[name]) ].index
    return aux_data_norm
