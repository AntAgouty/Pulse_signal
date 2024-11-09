import numpy as np
from scipy.signal import find_peaks, hilbert, savgol_filter, cwt, ricker, butter, filtfilt
from scipy.ndimage import median_filter, uniform_filter1d,zoom
from statsmodels.tsa.ar_model import AutoReg
from plotly import graph_objects as go
import pywt
from scipy.cluster.hierarchy import fclusterdata
from collections import Counter
import gc

class PulseDetector:
    def __init__(self, signal, sample_rate=50000):
        self.original_signal = signal  # Store the original signal for reference
        self.signal = signal
        self.sample_rate = sample_rate
        self.detection_results = {}
        self.colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 
            'lime', 'brown', 'pink', 'gold', 'darkblue', 'teal', 'darkgreen',
            'crimson', 'indigo', 'orchid', 'maroon', 'chocolate'
        ]

    # Robust Baseline Calculation Methods
    def calculate_baseline(self, method="savgol", window_size=101, polyorder=2):
        if method == "savgol":
            return savgol_filter(self.signal, window_length=window_size, polyorder=polyorder)
        elif method == "wavelet":
            coeffs = pywt.wavedec(self.signal, 'db4', level=1)
            coeffs[0] = np.zeros_like(coeffs[0])
            return pywt.waverec(coeffs, 'db4')
        elif method == "highpass":
            nyquist = 0.5 * self.sample_rate
            cutoff_freq = 5000
            normal_cutoff = cutoff_freq / nyquist
            b, a = butter(5, normal_cutoff, btype='high', analog=False)
            return filtfilt(b, a, self.signal)
        else:
            raise ValueError("Invalid baseline calculation method.")

    # Peak Detection Methods
    def hilbert_envelope_detection(self):
        analytic_signal = hilbert(self.signal)
        envelope = np.abs(analytic_signal)
        adaptive_threshold = np.median(envelope) + 2 * np.std(envelope)
        peaks, _ = find_peaks(envelope, height=adaptive_threshold)
        self.detection_results["Hilbert Envelope Detection"] = np.array(peaks, dtype=int)

    def autoregressive_model_residuals(self, lag=10):
        # Step 1: Use np.correlate to create a lightweight AR-like approximation
        # Using a sliding window for lagged predictions to approximate AR model behavior
        padded_signal = np.pad(self.signal, (lag, 0), 'constant', constant_values=0)
        approximations = np.correlate(padded_signal, np.ones(lag), mode='valid') / lag
        
        # Step 2: Calculate residuals by directly subtracting the lagged approximation
        residuals = self.signal[lag:] - approximations[:len(self.signal) - lag]
        
        # Step 3: Calculate adaptive threshold for peak detection
        adaptive_threshold = np.mean(residuals) + 3 * np.std(residuals)
        
        # Step 4: Detect peaks in the residuals
        peaks, _ = find_peaks(residuals, height=adaptive_threshold)
        self.detection_results["Autoregressive Model Residuals"] = np.array(peaks + lag, dtype=int)
            

    def wavelet_transform_detection(self, selected_scale=5, scale_range=3, downsample_factor=2):
        # Step 1: Downsample the signal to reduce computation if it's large
        downsampled_signal = self.signal[::downsample_factor]
        
        # Step 2: Calculate CWT on a limited range of widths around the selected_scale
        widths = np.arange(max(1, selected_scale - scale_range), selected_scale + scale_range + 1)
        cwt_matrix = cwt(downsampled_signal, ricker, widths)
        
        # Step 3: Select the wavelet coefficients at the desired scale
        wavelet_coefficients = cwt_matrix[widths == selected_scale][0]
        
        # Step 4: Resample back to original length if downsampling was applied
        wavelet_coefficients = zoom(wavelet_coefficients, downsample_factor, order=1) if downsample_factor > 1 else wavelet_coefficients
        
        # Step 5: Calculate the adaptive threshold and detect peaks
        adaptive_threshold = np.median(wavelet_coefficients) + 2 * np.std(wavelet_coefficients)
        peaks, _ = find_peaks(wavelet_coefficients, height=adaptive_threshold)
        
        # Step 6: Store both peak indices (x) and corresponding amplitudes (y values)
        peak_y_values = self.signal[peaks]  # get corresponding y values from original signal
        self.detection_results["Wavelet Transform Detection"] = (peaks, peak_y_values)

    def short_time_energy_detection(self, window_size=10, step_size=5):
        # Step 1: Create a rolling window view into the signal for the given window size
        shape = (len(self.signal) - window_size + 1, window_size)
        strides = (self.signal.strides[0], self.signal.strides[0])
        rolling_windows = np.lib.stride_tricks.as_strided(self.signal, shape=shape, strides=strides)
        
        # Step 2: Compute short-time energy by summing squares within each window
        ste = np.sum(rolling_windows ** 2, axis=1)[::step_size]

        # Step 3: Calculate adaptive threshold for peak detection
        adaptive_threshold = np.median(ste) + 2 * np.std(ste)
        
        # Step 4: Detect peaks in the short-time energy
        peaks, _ = find_peaks(ste, height=adaptive_threshold)
        
        # Step 5: Map peak indices back to the original signal indices
        self.detection_results["Short-Time Energy Detection"] = np.array(peaks * step_size + window_size // 2, dtype=int)

    def savitzky_golay_smoothing(self, window_length=11, polyorder=2):
        smoothed_signal = savgol_filter(self.signal, window_length=window_length, polyorder=polyorder)
        adaptive_threshold = np.median(smoothed_signal) + 2 * np.std(smoothed_signal)
        peaks, _ = find_peaks(smoothed_signal, height=adaptive_threshold)
        self.detection_results["Savitzky-Golay Smoothing"] = np.array(peaks, dtype=int)

    def teager_kaiser_energy_operator(self):
        tkeo_signal = self.signal[1:-1]**2 - self.signal[:-2] * self.signal[2:]
        adaptive_threshold = np.median(tkeo_signal) + 2 * np.std(tkeo_signal)
        peaks, _ = find_peaks(tkeo_signal, height=adaptive_threshold)
        self.detection_results["Teager-Kaiser Energy Operator"] = np.array(peaks + 1, dtype=int)

    def median_filter_gradient(self, window_size=5):
        filtered_signal = median_filter(self.signal, size=window_size)
        gradient_signal = np.gradient(filtered_signal)
        adaptive_threshold = np.median(gradient_signal) + 2 * np.std(gradient_signal)
        peaks, _ = find_peaks(gradient_signal, height=adaptive_threshold)
        self.detection_results["Median Filter with Gradient"] = np.array(peaks, dtype=int)

    def differential_detection(self):
        differential_signal = np.diff(self.signal)
        adaptive_threshold = np.median(np.abs(differential_signal)) + 2 * np.std(np.abs(differential_signal))
        peaks, _ = find_peaks(np.abs(differential_signal), height=adaptive_threshold)
        self.detection_results["Differential Detection"] = np.array(peaks, dtype=int)

    def cusum_detection(self):
        cusum_pos = np.maximum.accumulate(self.signal - 0.5)
        adaptive_threshold = np.median(cusum_pos) + 2 * np.std(cusum_pos)
        peaks, _ = find_peaks(cusum_pos, height=adaptive_threshold)
        self.detection_results["CUSUM Detection"] = np.array(peaks, dtype=int)

    def rms_energy_plateau(self, window_size=10, rms_threshold=0.1, amplitude_threshold=0.5):
        # Step 1: Compute cumulative sum of squares for efficient windowed RMS calculation
        squared_signal = self.signal ** 2
        cumsum_sq = np.cumsum(squared_signal)
        
        # Step 2: Calculate windowed RMS using cumulative sums for sliding windows
        rms_signal = np.sqrt((cumsum_sq[window_size:] - cumsum_sq[:-window_size]) / window_size)
        
        # Step 3: Extend rms_signal to the same length as the original signal
        rms_signal = np.concatenate((rms_signal, np.full(len(self.signal) - len(rms_signal), rms_signal[-1])))
        
        # Step 4: Apply threshold conditions
        plateau_mask = (rms_signal > rms_threshold) & (self.signal > amplitude_threshold)
        self.detection_results["RMS Energy Plateau"] = np.array(np.where(plateau_mask)[0], dtype=int)


    def median_filter_plateau(self, window_size=5, diff_threshold=0.05, amplitude_threshold=0.5):
        filtered_signal = median_filter(self.signal, size=window_size)
        differential_signal = np.abs(np.diff(filtered_signal))
        differential_signal = np.pad(differential_signal, (0, 1), mode='edge')
        plateau_mask = (differential_signal < diff_threshold) & (self.signal > amplitude_threshold)
        self.detection_results["Median Filter Plateau"] = np.array(np.where(plateau_mask)[0], dtype=int)

    def histogram_plateau(self, window_size=10, flatness_threshold=0.02, amplitude_threshold=0.5):
        # Use a smaller smoothing window to capture more local variation, closer to the histogram's effect
        smoothed_signal = uniform_filter1d(self.signal, size=window_size // 2)
        local_std = uniform_filter1d((self.signal - smoothed_signal) ** 2, size=window_size // 2)
        histogram_flatness = np.sqrt(local_std)

        # Pad to maintain length consistency
        histogram_flatness = np.pad(histogram_flatness, (0, len(self.signal) - len(histogram_flatness)), mode='edge')
        plateau_mask = (histogram_flatness < flatness_threshold) & (self.signal > amplitude_threshold)
        self.detection_results["Histogram Plateau"] = np.array(np.where(plateau_mask)[0], dtype=int)

    # Unified Filtering of Peaks and Plateaus based on Baseline
    def filter_peaks_above_baseline(self, baseline):
        baseline_deviation = self.signal - baseline
        threshold = np.median(baseline_deviation) + 3 * np.std(baseline_deviation)
        filtered_results = {}

        for method, detections in self.detection_results.items():
            # Ensure each detection point `p` is evaluated as a scalar.
            if isinstance(detections, np.ndarray):
                filtered_results[method] = np.array([p for p in detections if self.signal[int(p)] > baseline[int(p)] + threshold], dtype=int)
            else:
                # Handle cases where detections might not be an array
                filtered_results[method] = detections

        self.detection_results = filtered_results

  # Consensus methods (Frequency and Clustering Consensus)
    def frequency_consensus(self, threshold=3):
        all_detections = []
        for method, indices in self.detection_results.items():
            all_detections.extend(indices)

        count_dict = Counter(all_detections)
        consensus_points = [idx for idx, count in count_dict.items() if count >= threshold]
        self.detection_results["Frequency Consensus"] = np.array(consensus_points, dtype=int)

    def clustering_consensus(self, distance_threshold=5, chunk_size=1000):
        # Step 1: Deduplicate all detection points without using np.unique
        all_detections = np.concatenate(list(self.detection_results.values()))
        all_detections = np.sort(all_detections)
        all_detections = all_detections[np.insert(np.diff(all_detections) > 0, 0, True)]

        if len(all_detections) <= chunk_size:
            # If data is small enough, do a single clustering step
            clusters = fclusterdata(all_detections[:, None], t=distance_threshold, criterion='distance')
            consensus_points = [int(all_detections[i]) for i, c in enumerate(clusters) if clusters.tolist().count(c) > 1]
        else:
            # Process each chunk separately
            consensus_points = []
            for i in range(0, len(all_detections), chunk_size):
                chunk = all_detections[i:i + chunk_size]
                sub_clusters = fclusterdata(chunk[:, None], t=distance_threshold, criterion='distance')
                
                # Add points with multiple occurrences in the same sub-cluster to consensus points
                sub_cluster_counts = Counter(sub_clusters)
                for idx, cluster_id in enumerate(sub_clusters):
                    if sub_cluster_counts[cluster_id] > 1:
                        consensus_points.append(int(chunk[idx]))
                
                # Trigger garbage collection after processing each chunk
                gc.collect()

        # Convert final consensus points to a numpy array for storage
        self.detection_results["Clustering Consensus"] = np.array(consensus_points, dtype=int)

    # Average Calculation for Consensus Methods with Outlier Removal
    def calculate_average_per_pulse(self, consensus_key):
        """Calculate average x and y values for each pulse in the given consensus key and store in detection_results."""
        consensus_points = self.detection_results.get(consensus_key, [])
        average_values = []
        
        if len(consensus_points) == 0:
            self.detection_results[f"{consensus_key} Averages"] = (np.array([]), np.array([]))
            return

        # Group points into pulses if they are close together
        pulses = []
        current_pulse = [consensus_points[0]]

        for i in range(1, len(consensus_points)):
            if consensus_points[i] - consensus_points[i - 1] <= 1:
                current_pulse.append(consensus_points[i])
            else:
                pulses.append(current_pulse)
                current_pulse = [consensus_points[i]]
        pulses.append(current_pulse)  # Append the last pulse

        # Calculate average x and y for each pulse
        for pulse in pulses:
            avg_x = np.mean(pulse)
            avg_y = np.mean(self.signal[pulse])
            average_values.append((avg_x, avg_y))

        avg_x, avg_y = np.array(average_values).T
        self.detection_results[f"{consensus_key} Averages"] = (avg_x, avg_y)

    def plot_results(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.signal, mode='lines', name='Original Signal'))
        
        # Plot detected peaks for each detection method
        for method, peaks in self.detection_results.items():
            if "Averages" not in method and len(peaks) > 0:
                fig.add_trace(go.Scatter(
                    x=peaks, 
                    y=self.signal[peaks], 
                    mode='markers', 
                    name=f"{method} Peaks"
                ))

        # Check for and plot averages for "Frequency Consensus" if it exists
        if "Frequency Consensus Averages" in self.detection_results:
            avg_x, avg_y = self.detection_results["Frequency Consensus Averages"]
            fig.add_trace(go.Scatter(
                x=avg_x, 
                y=avg_y, 
                mode='markers+text', 
                marker=dict(color="black", size=10, symbol='x'), 
                name="Frequency Consensus Average"
            ))

        # Check for and plot averages for "Clustering Consensus" if it exists
        if "Clustering Consensus Averages" in self.detection_results:
            avg_x, avg_y = self.detection_results["Clustering Consensus Averages"]
            fig.add_trace(go.Scatter(
                x=avg_x, 
                y=avg_y, 
                mode='markers+text', 
                marker=dict(color="purple", size=10, symbol='x'), 
                name="Clustering Consensus Average"
            ))

        # Layout customization
        fig.update_layout(
            title="Pulse Detection and Consensus Results",
            xaxis_title="Sample Number",
            yaxis_title="Amplitude",
            template="plotly_white",
            hovermode="closest"
        )
        fig.show()

    
    def detect_all(self, baseline_method="savgol"):
        # Run all detection methods
        self.hilbert_envelope_detection()
        self.autoregressive_model_residuals()
        self.wavelet_transform_detection()
        self.short_time_energy_detection()
        self.savitzky_golay_smoothing()
        self.teager_kaiser_energy_operator()
        self.median_filter_gradient()
        self.differential_detection()
        self.cusum_detection()
        self.rms_energy_plateau()
        self.median_filter_plateau()
        self.histogram_plateau()

        # Baseline calculation and filter out peaks below it
        baseline = self.calculate_baseline(method=baseline_method)
        self.filter_peaks_above_baseline(baseline)

        # Run consensus methods
        self.frequency_consensus()
        self.clustering_consensus()

        # Calculate average values for each consensus method and store in detection_results
        self.calculate_average_per_pulse("Frequency Consensus")
        self.calculate_average_per_pulse("Clustering Consensus")
