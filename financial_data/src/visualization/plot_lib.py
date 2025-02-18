import plotly.express as px
import seaborn as sns

def trends_from_dataframe(timeline_df, 
                          **kwargs):

    # Plot return for each company
    fig = px.line(timeline_df, 
                x=timeline_df.index, 
                y=timeline_df.columns, 
                title=kwargs["title"],)
    
    # For each x-value show all the y-values simultaneously in the dynamic box
    fig.update_traces(hoverinfo='y')
    fig.update_layout(hovermode = "x unified")
    # Change y-axis name
    fig.update_yaxes(title_text="Return")
    # Show the plot
    fig.show()

def plot_log_return_difference(log_returns_difference, 
                               companies,
                               PATH_REPORTS_DIR=None,
                               savefig=False):
    # Plot log returns difference with color line purple
    fig = px.line(log_returns_difference, 
                  title=f"Log Returns Difference ({companies[0]} - {companies[1]})")
    # Plot horizontal line 
    fig.add_hline(y=0, line_dash="dot", line_color="red")
    # Set x-label
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Log Returns Difference')
    # Set line color to purple
    fig.update_traces(line_color='purple')
    # Show and save plot
    if savefig:
        fig.write_image(f"{PATH_REPORTS_DIR}/log_returns_difference_{companies[0]}_{companies[1]}.jpeg")
    fig.show()
    
def correlation_heatmap(returns_of_companies):
    corr = returns_of_companies.corr()
    fig_corr = px.imshow(corr, 
                        labels=dict(color="Correlation"),
                        color_continuous_scale='RdBu_r'
    )
    fig_corr.show()
    
def plot_scatter_returns(log_returns_difference, 
                         companies,
                         PATH_REPORTS_DIR=None,
                         savefig=False):
    
    xdata_label = companies[1]
    ydata_label = companies[0]
    fig = px.scatter(x=log_returns_difference[xdata_label], 
                    y=log_returns_difference[ydata_label])
    fig.update_xaxes(title_text=xdata_label)
    fig.update_yaxes(title_text=ydata_label)
    if savefig:
        fig.write_image(f"{PATH_REPORTS_DIR}/scatter_returns_{companies[0]}_{companies[1]}.jpeg")
    fig.show()