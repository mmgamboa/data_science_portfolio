import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from .show_report_raw_data import split_families_and_members


def preproc_data(dataset, 
                 age_nbins:int=50,
                 verbose=False):
    
    # Build Family distribution
    family_fig = plot_violin_distro_family(dataset, verbose=verbose)
    #cryo_fig = plot_cryo_sleep_distribution(dataset, verbose=verbose)
    #homeplanet_fig = plot_home_planet_distribution(dataset, verbose=verbose)
    #destinationplanet_fig = plot_destination_planet_distribution(dataset, verbose=verbose)
    ##age_fig = plot_age_distribution(dataset, nbins=age_nbins, verbose=verbose)
    
    #nan_fig = plot_nan_distribution(dataset, verbose=verbose)
    
    return (family_fig, )

def plot_violin_distro_family(dataset,
                             verbose=False):
    """
    Plot histogram of the family distribution with outlier criteria explicited
    dataset: pd.DataFrame
    
    Returns the figure of the family distribution
    """
    families, members = split_families_and_members(dataset)
    
    number_of_families = len(np.unique(families))
    
    if verbose:
        print(f"There are {number_of_families} families in the training data")
        
    members_with_nan = np.append(members, [np.nan] * dataset['PassengerId'].isna().sum())
    fig = px.histogram(members_with_nan, nbins=len(np.unique(members_with_nan)), title='Family sizes distribution')
    # Set 'Unknown' color to gray
    #fig.update_traces(marker=dict(colors=['gray' if name == 'Unknown' else None for name in fig.data[0].labels]))
    # Add title to legend
    #fig.update_layout(legend_title_text='Family members')

    #print(f"There are {train_dataset.isna().sum(axis=0).sum()} NaN values in the training data")

    return fig
