import plotly.graph_objects as go

def age_data(train, test,
             verbose=False):
    
    
    # Select age-related columns
    age_cols = [col for col in train.columns if col.startswith('GroupedAge_')]
    # Compute frequency counts
    age_group_counts = train[age_cols].sum()
    age_group_counts_test = test[age_cols].sum()

    #print(f"I am in the following module: {__name__}")
    
    # Create a Plotly bar chart
    fig = go.Figure()

    # Add training data bars
    fig.add_trace(go.Bar(
        x=age_group_counts.index, 
        y=age_group_counts.values, 
        name="Training Data", 
        marker_color='skyblue'
    ))

    # Add test data bars
    fig.add_trace(go.Bar(
        x=age_group_counts_test.index, 
        y=age_group_counts_test.values, 
        name="Test Data", 
        marker_color='salmon'
    ))

    # Update layout
    fig.update_layout(
        title="Frequency Histogram of Grouped Age",
        xaxis_title="Age Group",
        yaxis_title="Frequency",
        barmode="group",  # Groups bars side by side
        xaxis=dict(tickangle=45)  # Rotate x-axis labels
    )

    # Show plot
    #fig.show()

    return fig