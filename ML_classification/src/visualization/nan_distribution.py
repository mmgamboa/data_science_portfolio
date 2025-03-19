import plotly.express as px
import pandas as pd

def nan_data(dataset,
            verbose=False):
    """
    This function receives a dataset and returns a plotly figure with the NaN values distribution.
    
    Parameters
    ----------
    dataset : pandas.DataFrame
        The dataset to be analyzed.
    verbose : bool
        If True, print additional information.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        A plotly figure with the NaN values distribution
    """
    
    print("[Stage]: NaN -> Detecting ")    
    
    nan_counts = dataset.isna().sum(axis=0)
    if verbose: 
        print("NaN values", nan_counts)
        # are there values that intersect?
        # print(train_data[train_data.isna().any(axis=1)])
        total_count_nan = dataset.isna().sum().sum()
        intersect_count = (dataset.isna().sum(axis=1) > 1).sum()
        print(intersect_count , "/", total_count_nan, " rows has more than 1 NaN." )

    #fig = px.bar(pd.DataFrame(nan_counts), nbins=len(nan_counts), title='NaN Values Count in Each Column')
    # Convert to DataFrame for better plotting
    nan_df = pd.DataFrame({'Feature': nan_counts.index, 'NaN Count': nan_counts.values})

    # Use px.bar instead of histogram
    fig = px.bar(nan_df, x='Feature', y='NaN Count', 
                 title='NaN Values Count in Each Column')

    fig.update_xaxes(title='Feature', tickangle=270)  # Rotate labels for readability
    fig.update_yaxes(title='# NaN values')
    
    print("[Stage]: NaN -> Done. ")    
    
    return (fig,)