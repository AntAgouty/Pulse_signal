import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import Client
import os

def load_and_combine_parquet_files(directory):
    # Define file patterns for different result types
    result_patterns = {
        "napetost_all": "results_napetost_all_batch_",
        "napetost_wavelet": "results_napetost_wavelet_batch_",
        "tok_all": "results_tok_all_batch_",
        "tok_wavelet": "results_tok_wavelet_batch_"
    }

    combined_data = {}

    for result_type, pattern in result_patterns.items():
        # List all files matching the pattern for the result type
        files = [f for f in os.listdir(directory) if f.startswith(pattern) and f.endswith(".parquet")]
        full_paths = [os.path.join(directory, f) for f in sorted(files)]
        
        # Read each file, keeping the original index, and concatenate without resetting the index
        dfs = [pd.read_parquet(path) for path in full_paths]
        combined_data[result_type] = pd.concat(dfs, axis=0)

    return combined_data




def average_the_values(df,  x_col, y_col, averaged_col_name = "averaged", interval_duration_seconds = 30, sampling_frequency = 50000):

    interval_ticks = sampling_frequency * interval_duration_seconds  # 1.5 million samples

    # Assuming df is your DataFrame and it has the following structure
    # Index: measurement files (file names)
    # Columns: 'napetost_wave_x' (X values in ticks), 'napetost_wave_y' (Y values of peaks)

    # Define a function to assign each 'napetost_wave_x' value to a 30-second interval bin
    def assign_interval_bin(x):
        return (x // interval_ticks) * interval_ticks

    # Apply the function to create an interval column
    df['interval_bin'] = df[x_col].apply(assign_interval_bin)

    # Group by file name and interval bin, then calculate the mean of 'napetost_wave_y'
    average_df = df.groupby([df.index, 'interval_bin'])[y_col].mean().reset_index()

    # Rename columns for clarity
    average_df.rename(columns={y_col: averaged_col_name}, inplace=True)

    average_df['x_values'] = average_df.index

    average_df.set_index("level_0", drop = True, inplace = True)

    return average_df


def apply_kalman_filter(df, x_col, y_col, new_col_name="kalman_filtered"):
    """
    Applies a Kalman filter to smooth the specified y-values in the DataFrame and adds the results as a new column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    x_col (str): The name of the x-values column.
    y_col (str): The name of the y-values column to be filtered.
    new_col_name (str): The name of the new column to hold the Kalman filtered values.

    Returns:
    None: The function modifies the DataFrame in place.
    """
    # Initialize an empty list to store the smoothed data
    smoothed_data = []

    # Process each file separately
    for file_name in df.index.unique():
        file_data = df.loc[file_name]
        y_values = file_data[y_col].values

        # Initialize the Kalman Filter with the first y-value
        kf = KalmanFilter(initial_state_mean=y_values[0], n_dim_obs=1)
        kf = kf.em(y_values, n_iter=5)  # Train filter parameters using the data

        # Apply the Kalman filter to the y-values
        smoothed_state_means, _ = kf.smooth(y_values)

        # Add the smoothed values to the list
        smoothed_data.extend(smoothed_state_means.flatten())

    # Add the smoothed data as a new column to the DataFrame
    df[new_col_name] = smoothed_data



def apply_kalman_filter_to_chunk(df_chunk, y_col, new_col_name="kalman_filtered", n_iter=5):
    """
    Applies Kalman filter to each unique measurement within a DataFrame chunk.

    Parameters:
    df_chunk (pd.DataFrame): A chunk of the original DataFrame.
    y_col (str): Name of the y-values column to filter.
    new_col_name (str): Name of the new column to store the Kalman filtered values.
    n_iter (int): Number of EM iterations for Kalman filter training.

    Returns:
    pd.DataFrame: DataFrame chunk with an additional column of Kalman filtered y-values.
    """
    # Initialize an empty list to hold smoothed data for each measurement
    smoothed_data = []

    # Process each measurement (file) in the chunk
    for file_name in df_chunk.index.unique():
        file_data = df_chunk.loc[file_name].copy()
        y_values = file_data[y_col].values

        # Initialize and train Kalman filter
        kf = KalmanFilter(initial_state_mean=y_values[0], n_dim_obs=1)
        kf = kf.em(y_values, n_iter=n_iter)

        # Smooth the data
        smoothed_state_means, _ = kf.smooth(y_values)
        file_data[new_col_name] = smoothed_state_means.flatten()

        smoothed_data.append(file_data)

    # Concatenate the processed data for each measurement
    return pd.concat(smoothed_data)

def parallel_kalman_filter(df, y_col, new_col_name="kalman_filtered", npartitions=4, n_iter=2):
    """
    Applies Kalman filter in parallel to each unique measurement using Dask.

    Parameters:
    df (pd.DataFrame): Input DataFrame with each file/measurement as an index.
    y_col (str): Name of the y-values column to filter.
    new_col_name (str): Name of the new column to store the Kalman filtered values.
    npartitions (int): Number of partitions for Dask to parallelize the computation.
    n_iter (int): Number of EM iterations for Kalman filter training.

    Returns:
    pd.DataFrame: DataFrame with Kalman-filtered values added as a new column.
    """
    client = Client(n_workers=12, threads_per_worker=1, processes=True)  # Use processes for better parallelization

    # Convert to Dask DataFrame and set the number of partitions
    ddf = dd.from_pandas(df, npartitions=npartitions)

    # Define Dask delayed tasks for each partition
    delayed_results = [
        delayed(apply_kalman_filter_to_chunk)(chunk, y_col, new_col_name, n_iter)
        for chunk in ddf.to_delayed()
    ]

    # Compute all partitions in parallel and collect results
    results = compute(*delayed_results)

        # Shut down Dask client after computation
    client.shutdown()

    # Concatenate results into a single DataFrame
    return pd.concat(results, axis=0)