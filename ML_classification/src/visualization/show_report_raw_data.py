import numpy as np
import pandas as pd

import plotly.express as px

def split_families_and_members(dataset: pd.DataFrame):
    # Get the number of families in PassengerId: xxxx_yy means xxxx family and yy member
    aux_data = np.array(dataset['PassengerId'].apply(lambda x: np.array(x.split("_"), dtype=int)))
    families = np.vstack(aux_data).T[0]
    members = np.vstack(aux_data).T[1]
    return families, members

def inspect_data(train_dataset, 
                 verbose=False):
    
    # Build Family distribution
    family_fig = plot_family_distribution(train_dataset, verbose=verbose)
    homeplanet_fig = plot_home_planet_distribution(train_dataset, verbose=verbose)
    
    return family_fig, homeplanet_fig

def plot_family_distribution(dataset: pd.DataFrame,
                             verbose = False):
        # Get number of families
    families, members = split_families_and_members(dataset)
    
    number_of_families = len(np.unique(families))
    
    if verbose:
        print(f"There are {number_of_families} families in the training data")
        
    members_with_nan = np.append(members, [np.nan] * dataset['PassengerId'].isna().sum())
    fig = px.pie(values=np.unique(members_with_nan, return_counts=True)[1], 
			 names=np.unique(members_with_nan, return_counts=True)[0], 
			 title='Family sizes distribution')
    # Add title to legend
    fig.update_layout(legend_title_text='Family members')

    #print(f"There are {train_dataset.isna().sum(axis=0).sum()} NaN values in the training data")

    return fig

def plot_home_planet_distribution(dataset: pd.DataFrame, 
                                   verbose=False):
    # Pie chart for HomePlanet
    
    planets_with_nan = dataset['HomePlanet'].fillna('Unknown')
    
    fig = px.pie(values=planets_with_nan.value_counts().values, 
                 names=planets_with_nan.value_counts().index,
                 title="Home Planet distribution")
    fig.update_layout(title='Home Planet distribution')
    fig.update_layout(legend_title_text='Home Planet')
    
    return fig