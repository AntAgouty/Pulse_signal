import numpy as np
from scipy.signal import savgol_filter, hilbert, butter, filtfilt, find_peaks
from scipy.ndimage import median_filter
from plotly import graph_objects as go
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.ar_model import AutoReg

class PlatoDetector:
    def __init__(self, signal):
        self.signal = signal
        self.detection_results = {}
        self.colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 
            'lime', 'brown', 'pink', 'gold', 'darkblue', 'teal', 'darkgreen',
            'crimson', 'indigo', 'orchid', 'maroon', 'chocolate', 'navy'
        ]
    
    # 1. Kernel Density Estimation for Plateau Detection
    def kernel_density_plateau(self, bandwidth=5, gradient_threshold=0.01, amplitude_threshold=0.5):
        signal_reshaped = self.signal.reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(signal_reshaped)
        smoothed_signal = np.exp(kde.score_samples(signal_reshaped))
        gradient_signal = np.gradient(smoothed_signal)
        plateau_mask = (np.abs(gradient_signal) < gradient_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Kernel Density Plateau", plateau_mask)

    # 2. RMS (Root Mean Square) Energy Detection for Plateau
    def rms_energy_plateau(self, window_size=10, rms_threshold=0.1, amplitude_threshold=0.5):
        rms_signal = np.array([
            np.sqrt(np.mean(self.signal[i:i + window_size]**2)) 
            for i in range(len(self.signal) - window_size + 1)
        ])
        rms_signal = np.pad(rms_signal, (0, len(self.signal) - len(rms_signal)), mode='edge')
        plateau_mask = (rms_signal > rms_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("RMS Energy Plateau", plateau_mask)

    # 3. Median Filter with Flatness Detection
    def median_filter_plateau(self, window_size=5, diff_threshold=0.05, amplitude_threshold=0.5):
        filtered_signal = median_filter(self.signal, size=window_size)
        differential_signal = np.abs(np.diff(filtered_signal))
        differential_signal = np.pad(differential_signal, (0, 1), mode='edge')
        plateau_mask = (differential_signal < diff_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Median Filter Plateau", plateau_mask)

    # 4. Savitzky-Golay Filter with Flatness Detection
    def savitzky_golay_plateau(self, window_length=11, polyorder=2, slope_threshold=0.01, amplitude_threshold=0.5):
        smoothed_signal = savgol_filter(self.signal, window_length=window_length, polyorder=polyorder)
        slope_signal = np.gradient(smoothed_signal)
        plateau_mask = (np.abs(slope_signal) < slope_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Savitzky-Golay Plateau", plateau_mask)

    # 5. Moving Average Variance Plateau
    def moving_average_variance_plateau(self, window_size=10, variance_threshold=0.01, amplitude_threshold=0.5):
        moving_average = np.convolve(self.signal, np.ones(window_size) / window_size, mode='valid')
        moving_variance = np.convolve((self.signal - moving_average[:len(self.signal)])**2, 
                                      np.ones(window_size) / window_size, mode='valid')
        moving_variance = np.pad(moving_variance, (0, len(self.signal) - len(moving_variance)), mode='edge')
        plateau_mask = (moving_variance < variance_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Moving Average Variance Plateau", plateau_mask)

    # 6. Differential Plateau Detection
    def differential_plateau(self, diff_threshold=0.05, amplitude_threshold=0.5):
        differential_signal = np.abs(np.diff(self.signal))
        differential_signal = np.pad(differential_signal, (0, 1), mode='edge')
        plateau_mask = (differential_signal < diff_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Differential Plateau", plateau_mask)

    # 7. Low-Pass Filter with Moving Average
    def lowpass_moving_average_plateau(self, cutoff=0.1, window_size=10, amplitude_threshold=0.5):
        b, a = butter(1, cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, self.signal)
        moving_average = np.convolve(filtered_signal, np.ones(window_size) / window_size, mode='valid')
        moving_average = np.pad(moving_average, (0, len(self.signal) - len(moving_average)), mode='edge')
        plateau_mask = (self.signal > amplitude_threshold) & (np.abs(np.gradient(moving_average)) < 0.01)
        self._extract_plateaus("Lowpass Moving Average Plateau", plateau_mask)

    # 8. Hilbert Envelope Plateau Detection
    def hilbert_envelope_plateau(self, slope_threshold=0.01, amplitude_threshold=0.5):
        analytic_signal = hilbert(self.signal)
        envelope = np.abs(analytic_signal)
        slope_signal = np.gradient(envelope)
        plateau_mask = (np.abs(slope_signal) < slope_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Hilbert Envelope Plateau", plateau_mask)

    # 9. Local Entropy Plateau
    def entropy_plateau(self, window_size=10, entropy_threshold=0.5, amplitude_threshold=0.5):
        entropy_signal = np.array([
            -np.sum(np.histogram(self.signal[i:i + window_size], bins=5, density=True)[0] * 
                    np.log2(np.histogram(self.signal[i:i + window_size], bins=5, density=True)[0] + 1e-10))
            for i in range(len(self.signal) - window_size + 1)
        ])
        entropy_signal = np.pad(entropy_signal, (0, len(self.signal) - len(entropy_signal)), mode='edge')
        plateau_mask = (entropy_signal < entropy_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Local Entropy Plateau", plateau_mask)

    # 10. Autoregressive Model Residual Plateau Detection
    def autoregressive_plateau(self, lag=10, residual_threshold=0.05, amplitude_threshold=0.5):
        ar_model = AutoReg(self.signal, lags=lag).fit()
        residuals = np.abs(self.signal[lag:] - ar_model.predict(start=lag, end=len(self.signal)-1))
        residuals = np.pad(residuals, (lag, 0), mode='edge')
        plateau_mask = (residuals < residual_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Autoregressive Plateau", plateau_mask)

    # 11. Moving Median Slope Detection
    def moving_median_slope_plateau(self, window_size=10, slope_threshold=0.01, amplitude_threshold=0.5):
        moving_median = np.array([np.median(self.signal[i:i + window_size]) for i in range(len(self.signal) - window_size + 1)])
        moving_median = np.pad(moving_median, (0, len(self.signal) - len(moving_median)), mode='edge')
        slope_signal = np.gradient(moving_median)
        plateau_mask = (np.abs(slope_signal) < slope_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Moving Median Slope Plateau", plateau_mask)

    # 12. Weighted Moving Average Plateau Detection
    def weighted_moving_average_plateau(self, window_size=10, slope_threshold=0.01, amplitude_threshold=0.5):
        weights = np.arange(1, window_size + 1)
        weighted_avg = np.convolve(self.signal, weights / weights.sum(), mode='valid')
        weighted_avg = np.pad(weighted_avg, (0, len(self.signal) - len(weighted_avg)), mode='edge')
        plateau_mask = (np.abs(np.gradient(weighted_avg)) < slope_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Weighted Moving Average Plateau", plateau_mask)

    # 13. Gaussian Smoothing with Flatness Detection
    def gaussian_smoothing_plateau(self, window_size=10, sigma=2, slope_threshold=0.01, amplitude_threshold=0.5):
        from scipy.ndimage import gaussian_filter1d
        smoothed_signal = gaussian_filter1d(self.signal, sigma=sigma)
        slope_signal = np.gradient(smoothed_signal)
        plateau_mask = (np.abs(slope_signal) < slope_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Gaussian Smoothing Plateau", plateau_mask)

    # 14. Local Range Detection for Flat Regions
    def local_range_plateau(self, window_size=10, range_threshold=0.05, amplitude_threshold=0.5):
        local_range = np.array([np.max(self.signal[i:i + window_size]) - np.min(self.signal[i:i + window_size]) 
                                for i in range(len(self.signal) - window_size + 1)])
        local_range = np.pad(local_range, (0, len(self.signal) - len(local_range)), mode='edge')
        plateau_mask = (local_range < range_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Local Range Plateau", plateau_mask)

    # 15. Histogram-Based Plateau Detection
    def histogram_plateau(self, window_size=10, flatness_threshold=0.05, amplitude_threshold=0.5):
        histogram_flatness = np.array([
            np.std(np.histogram(self.signal[i:i + window_size], bins=5, density=True)[0])
            for i in range(len(self.signal) - window_size + 1)
        ])
        histogram_flatness = np.pad(histogram_flatness, (0, len(self.signal) - len(histogram_flatness)), mode='edge')
        plateau_mask = (histogram_flatness < flatness_threshold) & (self.signal > amplitude_threshold)
        self._extract_plateaus("Histogram Plateau", plateau_mask)

    def _extract_plateaus(self, method_name, plateau_mask):
        """Helper function to extract continuous plateau regions from a boolean mask."""
        plateaus = []
        start = None
        for i, is_plateau in enumerate(plateau_mask):
            if is_plateau and start is None:
                start = i
            elif not is_plateau and start is not None:
                if i - start > 1:
                    plateaus.append((start, i))
                start = None
        if start is not None:
            plateaus.append((start, len(self.signal)))
        self.detection_results[method_name] = plateaus

    def plot_results(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.signal, mode='lines', name='Original Signal'))
        
        for idx, (method, plateaus) in enumerate(self.detection_results.items()):
            x_values = []
            y_values = []
            
            for start, end in plateaus:
                x_values.extend(list(range(start, end)))
                y_values.extend(self.signal[start:end])
            
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='markers',
                marker=dict(color=self.colors[idx % len(self.colors)], size=6),
                name=f"{method}"
            ))

        fig.update_layout(
            title="Pulse Plateau Detection Results",
            xaxis_title="Sample Number",
            yaxis_title="Amplitude",
            template="plotly_white",
            hovermode="closest"
        )
        fig.show()
    
    def detect_all_plateaus(self):
        """Run all plateau detection methods and store results."""
        self.kernel_density_plateau()
        self.rms_energy_plateau()
        self.median_filter_plateau()
        self.savitzky_golay_plateau()
#        self.moving_average_variance_plateau()
        self.differential_plateau()
        self.lowpass_moving_average_plateau()
        self.hilbert_envelope_plateau()
        self.entropy_plateau()
        self.autoregressive_plateau()
        self.moving_median_slope_plateau()
        self.weighted_moving_average_plateau()
        self.gaussian_smoothing_plateau()
        self.local_range_plateau()
        self.histogram_plateau()
