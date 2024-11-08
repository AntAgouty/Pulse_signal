import os
import bioread
import pandas as pd
from Peaker import PulseDetector

# Define the directory containing the files
file_directory = "data"

od_s = 900
do_s = 1800

# Dictionaries to store results
results_tok_all_dict = {}
results_napetost_all_dict = {}
results_tok_wavelet_dict = {}
results_napetost_wavelet_dict = {}

def process_file(file_name):
    # Check if the file ends with '.acq'
    if not file_name.endswith('.acq'):
        return  # Skip non .acq files

    # Construct the full file path
    file_path = os.path.join(file_directory, file_name)

    # Read the .acq file using bioread
    data = bioread.read_file(file_path)
    napetost = None
    tok = None

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

            # Store results
            results_napetost_all_dict[file_name] = (napetost_avg_x, napetost_avg_y)
            results_napetost_wavelet_dict[file_name] = napetost_wave_y
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

            # Store results
            results_tok_all_dict[file_name] = (tok_avg_x, tok_avg_y)
            results_tok_wavelet_dict[file_name] = tok_wave_y
            del pulse_detector_tok  # Free memory

def save_results_to_csv():
    # Convert results_napetost_all_dict to DataFrame and save
    napetost_all_df = pd.DataFrame({
        file_name: {"napetost_x": x_val, "napetost_y": y_val}
        for file_name, (x_val, y_val) in results_napetost_all_dict.items()
    }).T.explode(['napetost_x', 'napetost_y'])
    napetost_all_df.to_csv("results_napetost_all_dict.csv", index_label="File")

    # Convert results_tok_all_dict to DataFrame and save
    tok_all_df = pd.DataFrame({
        file_name: {"tok_x": x_val, "tok_y": y_val}
        for file_name, (x_val, y_val) in results_tok_all_dict.items()
    }).T.explode(['tok_x', 'tok_y'])
    tok_all_df.to_csv("results_tok_all_dict.csv", index_label="File")

    # Convert results_napetost_wavelet_dict to DataFrame and save
    napetost_wavelet_df = pd.DataFrame({
        file_name: {"napetost_wave_y": y_val}
        for file_name, y_val in results_napetost_wavelet_dict.items()
    }).T.explode('napetost_wave_y')
    napetost_wavelet_df.to_csv("results_napetost_wavelet_dict.csv", index_label="File")

    # Convert results_tok_wavelet_dict to DataFrame and save
    tok_wavelet_df = pd.DataFrame({
        file_name: {"tok_wave_y": y_val}
        for file_name, y_val in results_tok_wavelet_dict.items()
    }).T.explode('tok_wave_y')
    tok_wavelet_df.to_csv("results_tok_wavelet_dict.csv", index_label="File")

def main():
    # Get list of all files in directory
    files = [f for f in os.listdir(file_directory) if f.endswith('.acq')]

    # Sequentially process each file without parallelization
    for file_name in files:
        process_file(file_name)

    # Save results to CSV files
    save_results_to_csv()

if __name__ == "__main__":
    main()
