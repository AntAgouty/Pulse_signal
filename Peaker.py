import numpy as np
from scipy.signal import find_peaks, hilbert, savgol_filter, cwt, ricker
from scipy.ndimage import median_filter
from sklearn.neighbors import KernelDensity
from statsmodels.tsa.ar_model import AutoReg
from plotly import graph_objects as go

class PulseDetector:
    def __init__(self, signal):
        self.signal = signal
        self.detection_results = {}
        self.colors = [
            'red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 
            'lime', 'brown', 'pink', 'gold', 'darkblue', 'teal', 'darkgreen',
            'crimson', 'indigo', 'orchid', 'maroon', 'chocolate'
        ]

    # 1. Hilbert Transform Envelope Detection
    def hilbert_envelope_detection(self, threshold_factor=1.5):
        analytic_signal = hilbert(self.signal)
        envelope = np.abs(analytic_signal)
        adaptive_threshold = threshold_factor * np.median(envelope)
        peaks, _ = find_peaks(envelope, height=adaptive_threshold)
        self.detection_results["Hilbert Envelope Detection"] = peaks

    # 2. Moving Variance Detection
    def moving_variance_detection(self, window_size=10, prominence=1):
        variance_signal = np.array([
            np.var(self.signal[i:i + window_size]) 
            for i in range(len(self.signal) - window_size + 1)
        ])
        peaks, _ = find_peaks(variance_signal, prominence=prominence)
        self.detection_results["Moving Variance Detection"] = peaks + window_size // 2

    # 3. Autoregressive Model Residual Analysis
    def autoregressive_model_residuals(self, lag=10, prominence=1):
        ar_model = AutoReg(self.signal, lags=lag).fit()
        residuals = self.signal[lag:] - ar_model.predict(start=lag, end=len(self.signal)-1)
        peaks, _ = find_peaks(residuals, prominence=prominence)
        self.detection_results["Autoregressive Model Residuals"] = peaks + lag
    
    # 4. Wavelet Transform Detection
    def wavelet_transform_detection(self, selected_scale=5, prominence=1):
        widths = np.arange(1, 20)
        cwt_matrix = cwt(self.signal, ricker, widths)
        wavelet_coefficients = cwt_matrix[selected_scale]
        peaks, _ = find_peaks(wavelet_coefficients, prominence=prominence)
        self.detection_results["Wavelet Transform Detection"] = peaks

    # 5. Short-Time Energy Detection
    def short_time_energy_detection(self, window_size=10, step_size=5, prominence=5):
        ste = np.array([
            np.sum(self.signal[i:i + window_size] ** 2) 
            for i in range(0, len(self.signal) - window_size, step_size)
        ])
        peaks, _ = find_peaks(ste, prominence=prominence)
        self.detection_results["Short-Time Energy Detection"] = peaks * step_size + window_size // 2

    # 6. Savitzky-Golay Smoothing
    def savitzky_golay_smoothing(self, window_length=11, polyorder=2, prominence=1):
        smoothed_signal = savgol_filter(self.signal, window_length=window_length, polyorder=polyorder)
        peaks, _ = find_peaks(smoothed_signal, prominence=prominence)
        self.detection_results["Savitzky-Golay Smoothing"] = peaks

    # 7. Teager-Kaiser Energy Operator
    def teager_kaiser_energy_operator(self, prominence=5):
        tkeo_signal = self.signal[1:-1]**2 - self.signal[:-2] * self.signal[2:]
        peaks, _ = find_peaks(tkeo_signal, prominence=prominence)
        self.detection_results["Teager-Kaiser Energy Operator"] = peaks + 1
    
    # 8. Kernel Density Estimation with Gradient Analysis
    def kernel_density_estimation(self, bandwidth=5, prominence=0.01):
        signal_reshaped = self.signal.reshape(-1, 1)
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(signal_reshaped)
        smoothed_signal = np.exp(kde.score_samples(signal_reshaped))
        gradient_signal = np.gradient(smoothed_signal)
        peaks, _ = find_peaks(gradient_signal, prominence=prominence)
        self.detection_results["Kernel Density Estimation"] = peaks

    # 9. Median Filtering with Gradient Detection
    def median_filter_gradient(self, window_size=5, prominence=1):
        filtered_signal = median_filter(self.signal, size=window_size)
        gradient_signal = np.gradient(filtered_signal)
        peaks, _ = find_peaks(gradient_signal, prominence=prominence)
        self.detection_results["Median Filter with Gradient"] = peaks

    # 10. Differential Signal Detection
    def differential_detection(self, prominence=1):
        differential_signal = np.diff(self.signal)
        peaks, _ = find_peaks(np.abs(differential_signal), prominence=prominence)
        self.detection_results["Differential Detection"] = peaks

    # 11. Local Entropy Detection
    def local_entropy_detection(self, window_size=10, prominence=1):
        entropy_signal = np.array([
            -np.sum(np.histogram(self.signal[i:i + window_size], bins=5, density=True)[0] * 
                    np.log2(np.histogram(self.signal[i:i + window_size], bins=5, density=True)[0] + 1e-10))
            for i in range(len(self.signal) - window_size + 1)
        ])
        peaks, _ = find_peaks(entropy_signal, prominence=prominence)
        self.detection_results["Local Entropy Detection"] = peaks

    # 12. RMS Energy Detection
    def rms_energy_detection(self, window_size=10, prominence=1):
        rms_signal = np.array([
            np.sqrt(np.mean(self.signal[i:i + window_size]**2)) 
            for i in range(len(self.signal) - window_size + 1)
        ])
        peaks, _ = find_peaks(rms_signal, prominence=prominence)
        self.detection_results["RMS Energy Detection"] = peaks + window_size // 2

    # 13. Moving Average Slope Detection
    def moving_average_slope(self, window_size=10, prominence=1):
        moving_average = np.convolve(self.signal, np.ones(window_size)/window_size, mode='valid')
        slope_signal = np.gradient(moving_average)
        peaks, _ = find_peaks(np.abs(slope_signal), prominence=prominence)
        self.detection_results["Moving Average Slope"] = peaks + window_size // 2

    # 14. Cumulative Sum (CUSUM) Detection
    def cusum_detection(self, threshold=1, drift=0.5):
        cusum_pos = np.maximum.accumulate(self.signal - drift)
        peaks, _ = find_peaks(cusum_pos, prominence=threshold)
        self.detection_results["CUSUM Detection"] = peaks


    # Plotting Results
    def plot_results(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.signal, mode='lines', name='Original Signal'))
        
        for idx, (method, peaks) in enumerate(self.detection_results.items()):
            fig.add_trace(go.Scatter(
                x=peaks,
                y=self.signal[peaks],
                mode='markers',
                marker=dict(color=self.colors[idx % len(self.colors)], size=8),
                name=f"{method} Peaks"
            ))

        fig.update_layout(
            title="Pulse Detection Results",
            xaxis_title="Sample Number",
            yaxis_title="Amplitude",
            template="plotly_white",
            hovermode="closest"
        )
        fig.show()
    
    # Running All Detection Methods
    def detect_all(self):
        self.hilbert_envelope_detection()
        self.moving_variance_detection()
        self.autoregressive_model_residuals()
        self.wavelet_transform_detection()
        self.short_time_energy_detection()
        self.savitzky_golay_smoothing()
        self.teager_kaiser_energy_operator()
        self.kernel_density_estimation()
        self.median_filter_gradient()
        self.differential_detection()
        self.local_entropy_detection()
        self.rms_energy_detection()
        self.moving_average_slope()
        self.cusum_detection()
