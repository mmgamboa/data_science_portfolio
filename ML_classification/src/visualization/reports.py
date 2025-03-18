from .show_report_raw_data import inspect_data
from .show_report_preprocessed_data import preproc_data
from .feature_display import age_data
from .nan_distribution import nan_data

def input_data_figs(dataset,
                    age_nbins=50,
                    verbose=False):
    """
    Generate figures for the input data
    dataset: pd.DataFrame
    age_nbins: int
    """
    return inspect_data(dataset, age_nbins=age_nbins, verbose=verbose)

def preproc_data_figs(dataset,
                      age_nbins:int=50,
                      verbose=False):
    """
    Generate figures for the preprocessed data
    dataset: pd.DataFrame
    age_nbins: int
    """
    return preproc_data(dataset)

def nan_fig(dataset,
            verbose=False):
    """
    Plot the NaN distribution
    dataset: pd.DataFrame
    """
    return nan_data(dataset, verbose)

def preproc_age_fig(train, test,
                    verbose=False):
    """
    Plot age distribution to see uniformity
    dataset: pd.DataFrame
    """
    return age_data(train, test, verbose=verbose)
    
#def feature_figs(dataset,
#                 verbose=False):
#    """
#    Plot dynamically feature-feature relations
#    dataset: pd.DataFrame
#    """
#    return feature_data(dataset, 
#                        verbose=verbose)
