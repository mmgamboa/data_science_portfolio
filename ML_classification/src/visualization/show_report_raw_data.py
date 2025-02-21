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
                 age_nbins:int=50,
                 verbose=False):
    
    # Build Family distribution
    family_fig = plot_family_distribution(train_dataset, verbose=verbose)
    cryo_fig = plot_cryo_sleep_distribution(train_dataset, verbose=verbose)
    homeplanet_fig = plot_home_planet_distribution(train_dataset, verbose=verbose)
    destinationplanet_fig = plot_destination_planet_distribution(train_dataset, verbose=verbose)
    age_fig = plot_age_distribution(train_dataset, nbins=age_nbins, verbose=verbose)
    
    return (family_fig, cryo_fig, homeplanet_fig,
            destinationplanet_fig, age_fig)

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
    # Set 'Unknown' color to gray
    fig.update_traces(marker=dict(colors=['gray' if name == 'Unknown' else None for name in fig.data[0].labels]))
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
    fig.update_traces(marker=dict(colors=['gray' if name == 'Unknown' else None for name in fig.data[0].labels]))
    
    fig.update_layout(legend_title_text='Home Planet')
    
    return fig

def plot_destination_planet_distribution(dataset: pd.DataFrame, 
                                   verbose=False):
    # Pie chart for HomePlanet
    
    planets_with_nan = dataset['Destination'].fillna('Unknown')
    
    fig = px.pie(values=planets_with_nan.value_counts().values, 
                 names=planets_with_nan.value_counts().index,
                 title="Destination Planet distribution")
    fig.update_layout(title='Destination Planet distribution')
    fig.update_traces(marker=dict(colors=['gray' if name == 'Unknown' else None for name in fig.data[0].labels]))
    
    fig.update_layout(legend_title_text='Destination Planet')
    
    return fig

def plot_cryo_sleep_distribution(dataset: pd.DataFrame, 
                                   verbose=False):
    # Pie chart for HomePlanet
    
    sleep_with_nan = dataset['CryoSleep'].fillna('Unknown')
    
    fig = px.pie(values=sleep_with_nan.value_counts().values, 
                 names=sleep_with_nan.value_counts().index,
                 title="Cryo Sleep distribution")
    fig.update_layout(title='Cryo Sleep distribution')
    fig.update_traces(marker=dict(colors=['gray' if name == 'Unknown' else None for name in fig.data[0].labels]))
    
    fig.update_layout(legend_title_text='Cryo Sleep')
    
    return fig

def plot_age_distribution(dataset: pd.DataFrame,
                          nbins: int=50,
                          verbose=False):
    
    fig = px.histogram(dataset, x="Age", color="Transported", nbins=nbins, title="Age Distribution")
    # Print inside the histogram the bins used in fig
    #fig.add_annotation(
    #    text=f"Bins used: {nbins}",  # Display bin count
    #    font=dict(color="black"),
    #    # delet arrow
    #    showarrow=False,
    #    )
    fig.update_layout(title='Age distribution - Bins used: {}'.format(nbins))
    fig.update_xaxes(title='Age')
    fig.update_yaxes(title='Count')
    
    return fig