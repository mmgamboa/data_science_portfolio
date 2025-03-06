import numpy as np
import pandas as pd

import plotly.express as px
from .show_report_raw_data import split_families_and_members


def inspect_data(dataset, 
                 age_nbins:int=50,
                 verbose=False):
    
    # Build Family distribution
    family_fig = plot_family_distribution(dataset, verbose=verbose)
    cryo_fig = plot_cryo_sleep_distribution(dataset, verbose=verbose)
    homeplanet_fig = plot_home_planet_distribution(dataset, verbose=verbose)
    destinationplanet_fig = plot_destination_planet_distribution(dataset, verbose=verbose)
    #age_fig = plot_age_distribution(dataset, nbins=age_nbins, verbose=verbose)
    
    #nan_fig = plot_nan_distribution(dataset, verbose=verbose)
    
    return (family_fig, cryo_fig, homeplanet_fig,
            destinationplanet_fig)#, age_fig, nan_fig)

def plot_family_distribution(dataset,
                             verbose=False):
    """
    Plot histogram of the family distribution with outlier criteria explicited
    dataset: pd.DataFrame
    
    Returns the figure of the family distribution
    """
    
    
    return

def plot_cryo_sleep_distribution(dataset,
                                verbose=False):
    """
    Plot histogram of the CryoSleep distribution with outlier criteria explicited
    dataset: pd.DataFrame
    
    Returns the figure of the CryoSleep distribution
    """
    
    return  

def plot_home_planet_distribution(dataset, 
                                   verbose=False):
    """
    Plot histogram of the HomePlanet distribution with outlier criteria explicited
    dataset: pd.DataFrame
    
    Returns the figure of the HomePlanet distribution
    """
    
    return    

def plot_destination_planet_distribution(dataset,
                                        verbose=False):
    """
    Plot histogram of the DestinationPlanet distribution with outlier criteria explicited
    dataset: pd.DataFrame
    
    Returns the figure of the DestinationPlanet distribution
    """
    
    return