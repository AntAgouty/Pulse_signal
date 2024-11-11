import plotly.graph_objects as go

def all_plot(df, x_val_col, y_val_col, first_x_measurments = -1):
# Select the first 10 unique files
    first_10_files = df.index.unique()
    filtered_df = df[df.index.isin(first_10_files)]

    # Create a Plotly figure
    fig = go.Figure()

    # Add a trace for each measurement file
    for file_name in filtered_df.index.unique():
        file_data = filtered_df[filtered_df.index == file_name]
        fig.add_trace(go.Scatter(
            x=file_data[x_val_col],
            y=file_data[y_val_col],
            mode='lines+markers',
            name=file_name
        ))

    # Update layout for readability and height
    fig.update_layout(
        title="30-Second Average Y-Values for First 10 Measurements",
        xaxis_title="Interval Bin (ticks)",
        yaxis_title="30-Second Average Y-Value",
        template="plotly_white",
        legend_title="Measurement Files",
        height=1500  # Increase height for a tall plot
    )

    # Show the plot
    fig.show()
