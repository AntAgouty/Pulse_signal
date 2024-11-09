import os
import bioread
import pandas as pd
from multiprocessing import Pool, Manager
from Peaker import PulseDetector

# Define the directory containing the files
file_directory = "data"

od_s = 900
do_s = 1800
batch_size = 100  # Number of files per batch when saving to Parquet
nu_threads = 6

def process_file(file_name, unprocessed_files):
    # Initialize dictionaries to store results for this file
    file_results = {
        "results_tok_all": None,
        "results_napetost_all": None,
        "results_tok_wavelet": None,
        "results_napetost_wavelet": None
    }
    
    # Check if the file ends with '.acq'
    if not file_name.endswith('.acq'):
        return file_name, file_results  # Return empty results for non-matching files

    # Construct the full file path
    file_path = os.path.join(file_directory, file_name)

    try:
        # Read the .acq file using bioread
        data = bioread.read_file(file_path)

        # Iterate through channels and process only necessary data
        for channel in data.channels:
            if channel.name == 'Napetost':
                time_intervals = channel.time_index[1] - channel.time_index[0]
                sample_rate = int(round(1.0 / time_intervals, 2))

                # Check if the signal segment is long enough for processing
                if len(channel.data) < int(do_s * sample_rate):
                    raise ValueError("File is too short for analysis")

                # Slice and limit data loading to necessary segments
                napetost = channel.data[int(od_s * sample_rate):int(do_s * sample_rate)]
                
                # Detect pulses in 'Napetost'
                print(f"extracting values for napetost in file {file_name}")
                pulse_detector_napetost = PulseDetector(napetost)
                pulse_detector_napetost.detect_all(baseline_method="savgol")
                napetost_avg_x, napetost_avg_y = pulse_detector_napetost.detection_results["Clustering Consensus Averages"]
                
                # Get both x and y for Wavelet Transform Detection
                napetost_wave_x, napetost_wave_y = pulse_detector_napetost.detection_results["Wavelet Transform Detection"]

                # Store results in the file-specific dictionary
                file_results["results_napetost_all"] = (napetost_avg_x, napetost_avg_y)
                file_results["results_napetost_wavelet"] = (napetost_wave_x, napetost_wave_y)  # Store both x and y values

                del pulse_detector_napetost  # Free memory

            elif channel.name == 'Tok':
                time_intervals = channel.time_index[1] - channel.time_index[0]
                sample_rate = int(round(1.0 / time_intervals, 2))

                # Check if the signal segment is long enough for processing
                if len(channel.data) < int(do_s * sample_rate):
                    raise ValueError("File is too short for analysis")

                tok = -channel.data[int(od_s * sample_rate):int(do_s * sample_rate)]
                
                # Detect pulses in 'Tok'
                print(f"extracting values for tok in file {file_name}")
                pulse_detector_tok = PulseDetector(tok)
                pulse_detector_tok.detect_all(baseline_method="savgol")
                tok_avg_x, tok_avg_y = pulse_detector_tok.detection_results["Clustering Consensus Averages"]
                
                # Get both x and y for Wavelet Transform Detection
                tok_wave_x, tok_wave_y = pulse_detector_tok.detection_results["Wavelet Transform Detection"]

                # Store results in the file-specific dictionary
                file_results["results_tok_all"] = (tok_avg_x, tok_avg_y)
                file_results["results_tok_wavelet"] = (tok_wave_x, tok_wave_y)  # Store both x and y values

                del pulse_detector_tok  # Free memory

    except ValueError as e:
        print(f"Could not analyze file {file_name}: {e}")
        unprocessed_files.append(file_name)  # Track unprocessed files
    except Exception as e:
        print(f"An error occurred with file {file_name}: {e}")
        unprocessed_files.append(file_name)  # Track unprocessed files

    return file_name, file_results


def save_results_to_parquet(results, batch_number):
    # Convert results to separate DataFrames and save in Parquet format
    results_napetost_all_dict = {file_name: res["results_napetost_all"] for file_name, res in results.items() if res["results_napetost_all"] is not None}
    results_tok_all_dict = {file_name: res["results_tok_all"] for file_name, res in results.items() if res["results_tok_all"] is not None}
    results_napetost_wavelet_dict = {file_name: res["results_napetost_wavelet"] for file_name, res in results.items() if res["results_napetost_wavelet"] is not None and len(res["results_napetost_wavelet"]) > 0}
    results_tok_wavelet_dict = {file_name: res["results_tok_wavelet"] for file_name, res in results.items() if res["results_tok_wavelet"] is not None and len(res["results_tok_wavelet"]) > 0}

    if results_napetost_all_dict:
        napetost_all_df = pd.DataFrame({
            file_name: {"napetost_x": x_val, "napetost_y": y_val}
            for file_name, (x_val, y_val) in results_napetost_all_dict.items()
        }).T.explode(['napetost_x', 'napetost_y'])
        napetost_all_df.to_parquet(f"results_napetost_all_batch_{batch_number}.parquet")

    if results_tok_all_dict:
        tok_all_df = pd.DataFrame({
            file_name: {"tok_x": x_val, "tok_y": y_val}
            for file_name, (x_val, y_val) in results_tok_all_dict.items()
        }).T.explode(['tok_x', 'tok_y'])
        tok_all_df.to_parquet(f"results_tok_all_batch_{batch_number}.parquet")

    if results_napetost_wavelet_dict:
        napetost_wavelet_df = pd.DataFrame({
            file_name: {"napetost_wave_x": x_val, "napetost_wave_y": y_val}
            for file_name, (x_val, y_val) in results_napetost_wavelet_dict.items()
        }).T.explode(['napetost_wave_x', 'napetost_wave_y'])
        napetost_wavelet_df.to_parquet(f"results_napetost_wavelet_batch_{batch_number}.parquet")

    if results_tok_wavelet_dict:
        tok_wavelet_df = pd.DataFrame({
            file_name: {"tok_wave_x": x_val, "tok_wave_y": y_val}
            for file_name, (x_val, y_val) in results_tok_wavelet_dict.items()
        }).T.explode(['tok_wave_x', 'tok_wave_y'])
        tok_wavelet_df.to_parquet(f"results_tok_wavelet_batch_{batch_number}.parquet")

def main():
    # Get list of all files in directory
    files = [f for f in os.listdir(file_directory) if f.endswith('.acq')]

    # Shared list to track files that couldn't be processed
    manager = Manager()
    unprocessed_files = manager.list()

    # Use multiprocessing pool limited to 4 processes to process files in parallel
    with Pool(processes=nu_threads) as pool:
        # Pass `unprocessed_files` to each process so it can record failures
        results = dict(pool.starmap(process_file, [(file, unprocessed_files) for file in files]))

    # Save results in batches to Parquet
    batch_number = 1
    results_batch = {}
    for i, (file_name, result) in enumerate(results.items()):
        results_batch[file_name] = result
        # Save batch when batch_size is reached
        if (i + 1) % batch_size == 0 or i + 1 == len(results):
            save_results_to_parquet(results_batch, batch_number)
            results_batch.clear()  # Clear the batch dictionary for the next set
            batch_number += 1

    # Print summary of unprocessed files
    if unprocessed_files:
        print("\nFiles that could not be analyzed:")
        for file_name in unprocessed_files:
            print(file_name)
    else:
        print("\nAll files were processed successfully.")

if __name__ == "__main__":
    main()
