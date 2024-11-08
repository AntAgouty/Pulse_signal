import os
import bioread
import pandas as pd
from multiprocessing import Pool
from Peaker import PulseDetector

# Define the directory containing the files
file_directory = "data"

od_s = 900
do_s = 1800

def process_file(file_name):
    # Initialize dictionaries to store results for this file
    file_results = {
        "results_tok_all": None,
        "results_napetost_all": None,
        "results_tok_wavelet": None,
        "results_napetost_wavelet": None
    }
    
    # Check if the file ends with '.acq'
    if not file_name.endswith('.acq'):
        return file_results  # Return empty results for non-matching files

    # Construct the full file path
    file_path = os.path.join(file_directory, file_name)

    # Read the .acq file using bioread
    data = bioread.read_file(file_path)

    # Iterate through channels and process only necessary data
    for channel in data.channels:
        if channel.name == 'Napetost':
            time_intervals = channel.time_index[1] - channel.time_index[0]
            sample_rate = int(round(1.0 / time_intervals, 2))

            # Slice and limit data loading to necessary segments
            napetost = channel.data[int(od_s * sample_rate):int(do_s * sample_rate)]
            
            # Detect pulses in 'Napetost'
            print(f"extracting values for napetost in file {file_name}")
            pulse_detector_napetost = PulseDetector(napetost)
            pulse_detector_napetost.detect_all(baseline_method="savgol")
            napetost_avg_x, napetost_avg_y = pulse_detector_napetost.detection_results["Clustering Consensus Averages"]
            napetost_wave_y = pulse_detector_napetost.detection_results["Wavelet Transform Detection"]

            # Store results in the file-specific dictionary
            file_results["results_napetost_all"] = (napetost_avg_x, napetost_avg_y)
            file_results["results_napetost_wavelet"] = napetost_wave_y
            del pulse_detector_napetost  # Free memory

        elif channel.name == 'Tok':
            time_intervals = channel.time_index[1] - channel.time_index[0]
            sample_rate = int(round(1.0 / time_intervals, 2))

            tok = -channel.data[int(od_s * sample_rate):int(do_s * sample_rate)]
            
            # Detect pulses in 'Tok'
            print(f"extracting values for tok in file {file_name}")
            pulse_detector_tok = PulseDetector(tok)
            pulse_detector_tok.detect_all(baseline_method="savgol")
            tok_avg_x, tok_avg_y = pulse_detector_tok.detection_results["Clustering Consensus Averages"]
            tok_wave_y = pulse_detector_tok.detection_results["Wavelet Transform Detection"]

            # Store results in the file-specific dictionary
            file_results["results_tok_all"] = (tok_avg_x, tok_avg_y)
            file_results["results_tok_wavelet"] = tok_wave_y
            del pulse_detector_tok  # Free memory

    return (file_name, file_results)

def save_results_to_csv(results):
    # Convert results to separate dictionaries
    results_napetost_all_dict = {file_name: res["results_napetost_all"] for file_name, res in results.items() if res["results_napetost_all"]}
    results_tok_all_dict = {file_name: res["results_tok_all"] for file_name, res in results.items() if res["results_tok_all"]}
    results_napetost_wavelet_dict = {file_name: res["results_napetost_wavelet"] for file_name, res in results.items() if res["results_napetost_wavelet"]}
    results_tok_wavelet_dict = {file_name: res["results_tok_wavelet"] for file_name, res in results.items() if res["results_tok_wavelet"]}

    # Save each result dictionary to a CSV
    if results_napetost_all_dict:
        napetost_all_df = pd.DataFrame({
            file_name: {"napetost_x": x_val, "napetost_y": y_val}
            for file_name, (x_val, y_val) in results_napetost_all_dict.items()
        }).T.explode(['napetost_x', 'napetost_y'])
        napetost_all_df.to_csv("results_napetost_all_dict.csv", index_label="File")

    if results_tok_all_dict:
        tok_all_df = pd.DataFrame({
            file_name: {"tok_x": x_val, "tok_y": y_val}
            for file_name, (x_val, y_val) in results_tok_all_dict.items()
        }).T.explode(['tok_x', 'tok_y'])
        tok_all_df.to_csv("results_tok_all_dict.csv", index_label="File")

    if results_napetost_wavelet_dict:
        napetost_wavelet_df = pd.DataFrame({
            file_name: {"napetost_wave_y": y_val}
            for file_name, y_val in results_napetost_wavelet_dict.items()
        }).T.explode('napetost_wave_y')
        napetost_wavelet_df.to_csv("results_napetost_wavelet_dict.csv", index_label="File")

    if results_tok_wavelet_dict:
        tok_wavelet_df = pd.DataFrame({
            file_name: {"tok_wave_y": y_val}
            for file_name, y_val in results_tok_wavelet_dict.items()
        }).T.explode('tok_wave_y')
        tok_wavelet_df.to_csv("results_tok_wavelet_dict.csv", index_label="File")

def main():
    # Get list of all files in directory
    files = [f for f in os.listdir(file_directory) if f.endswith('.acq')]

    # Use multiprocessing pool to process files in parallel
    with Pool() as pool:
        results = dict(pool.map(process_file, files))

    # Save results to CSV files
    save_results_to_csv(results)

if __name__ == "__main__":
    main()
