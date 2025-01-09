# mfc/mfc/goal_pattern_detector/time_window_manager.py

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import correlate
from sklearn.metrics import mutual_info_score
from statsmodels.tsa.stattools import grangercausalitytests
import multiprocessing
from datetime import timedelta
import logging

class TimeWindowManager:
    def __init__(self, fixed_size=100, overlap_ratio=0.5, max_lag=5):
        """
        Initializes the TimeWindowManager with fixed and adaptive windowing capabilities.

        Args:
            fixed_size (int): Size of each fixed window.
            overlap_ratio (float): Ratio of overlap between consecutive windows.
            max_lag (int): Maximum lag for Granger causality.
        """
        self.fixed_size = fixed_size
        self.overlap_ratio = overlap_ratio
        self.step_size = int(fixed_size * (1 - overlap_ratio))
        self.max_lag = max_lag
        self.trigger_events = {'pattern_detected'}

        logging.info(f"Initialized TimeWindowManager with fixed_size={fixed_size}, overlap_ratio={overlap_ratio}")

    def create_time_windows(self, data: Dict[str, List[int]], adaptive: bool = False, triggers: Optional[List[int]] = None) -> List[Dict[str, List[int]]]:
        """
        Creates time windows from the data based on fixed or adaptive strategies.

        Args:
            data (dict): Dictionary of CA states where keys are CA IDs and values are their state sequences.
            adaptive (bool): Whether to create adaptive windows based on triggers.
            triggers (list): List of trigger events to adjust windowing.

        Returns:
            list: List of window data subsets.
        """
        windows = []
        start = 0
        data_length = len(next(iter(data.values())))  # Assuming all CAs have the same length

        while start < data_length:
            if adaptive and triggers and start in triggers:
                window_size = self.determine_window_size(data, start)
                logging.info(f"Adaptive windowing triggered at step {start}. Window size set to {window_size}.")
            else:
                window_size = self.fixed_size

            end = start + window_size
            window = {ca_id: states[start:end] for ca_id, states in data.items()}
            windows.append(window)
            logging.debug(f"Created window from {start} to {end}.")

            start += self.step_size

        logging.info(f"Total windows created: {len(windows)}")
        return windows

    def determine_window_size(self, data: Dict[str, List[int]], start: int) -> int:
        """
        Determines the window size based on the trigger event.

        Args:
            data (dict): Dictionary of CA states.
            start (int): Start index of the window.

        Returns:
            int: Determined window size.
        """
        # Enhanced dynamic window size determination based on specific trigger characteristics
        # Example: Analyze recent patterns or metrics to decide window size
        # For simplicity, doubling the fixed size
        window_size = self.fixed_size * 2
        logging.debug(f"Determined adaptive window size: {window_size}")
        return window_size

    def find_correlated_cas(self, window_data: Dict[str, List[int]], methods: List[str] = ['cross_correlation', 'mutual_information', 'granger_causality'], threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Finds correlated CAs within a window using the specified correlation measures.

        Args:
            window_data (dict): Dictionary where keys are CA IDs and values are their state sequences.
            methods (list): List of correlation measures to use.
            threshold (float): Threshold above which CAs are considered correlated.

        Returns:
            list: List of tuples representing correlated CA pairs and their correlation scores.
        """
        correlated_pairs = []
        cas = list(window_data.keys())
        pairs = [(cas[i], cas[j]) for i in range(len(cas)) for j in range(i+1, len(cas))]

        def process_pair(pair):
            ca1, ca2 = pair
            scores = {}
            for method in methods:
                if method == 'cross_correlation':
                    scores['cross_correlation'] = self.calculate_cross_correlation(window_data[ca1], window_data[ca2])
                elif method == 'mutual_information':
                    scores['mutual_information'] = self.calculate_mutual_information(window_data[ca1], window_data[ca2])
                elif method == 'granger_causality':
                    scores['granger_causality'] = self.calculate_granger_causality(window_data[ca1], window_data[ca2])
            # Aggregate scores (example: average)
            if scores:
                aggregated_score = np.mean(list(scores.values()))
                if aggregated_score >= threshold:
                    return (ca1, ca2, aggregated_score)
            return None

        with multiprocessing.Pool() as pool:
            results = pool.map(process_pair, pairs)

        for result in results:
            if result:
                correlated_pairs.append(result)

        logging.info(f"Found {len(correlated_pairs)} correlated CA pairs in current window.")
        return correlated_pairs

    def calculate_cross_correlation(self, ca1: List[int], ca2: List[int]) -> float:
        """
        Calculates the cross-correlation between two CAs.

        Args:
            ca1 (list or np.ndarray): State sequence of CA1.
            ca2 (list or np.ndarray): State sequence of CA2.

        Returns:
            float: Normalized cross-correlation coefficient.
        """
        if len(ca1) != len(ca2) or len(ca1) == 0:
            logging.warning("CAs have unequal lengths or are empty for cross-correlation.")
            return 0.0
        correlation = np.corrcoef(ca1, ca2)[0, 1]
        logging.debug(f"Cross-correlation between {ca1} and {ca2}: {correlation}")
        return correlation

    def calculate_mutual_information(self, ca1: List[int], ca2: List[int], bins: int = 10) -> float:
        """
        Calculates the mutual information between two CAs.

        Args:
            ca1 (list or np.ndarray): State sequence of CA1.
            ca2 (list or np.ndarray): State sequence of CA2.
            bins (int): Number of bins for discretization.

        Returns:
            float: Mutual information score.
        """
        if len(ca1) != len(ca2) or len(ca1) == 0:
            logging.warning("CAs have unequal lengths or are empty for mutual information.")
            return 0.0
        # Discretize the continuous variables
        ca1_binned = np.digitize(ca1, bins=np.linspace(min(ca1), max(ca1), bins))
        ca2_binned = np.digitize(ca2, bins=np.linspace(min(ca2), max(ca2), bins))
        mutual_info = mutual_info_score(ca1_binned, ca2_binned)
        logging.debug(f"Mutual Information between CA1 and CA2: {mutual_info}")
        return mutual_info

    def calculate_granger_causality(self, ca1: List[int], ca2: List[int]) -> float:
        """
        Calculates the Granger causality p-value between two CAs.

        Args:
            ca1 (list or np.ndarray): State sequence of CA1.
            ca2 (list or np.ndarray): State sequence of CA2.

        Returns:
            float: Minimum p-value across lags.
        """
        if len(ca1) < self.max_lag + 1 or len(ca2) < self.max_lag + 1:
            logging.warning("Not enough data points for Granger causality test.")
            return 1.0  # Non-significant
        data = np.vstack([ca2, ca1]).T  # CA2 causes CA1
        try:
            test = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
            p_values = [round(test[i+1][0]['ssr_ftest'][1], 4) for i in range(self.max_lag)]
            min_p_value = min(p_values)
            logging.debug(f"Granger causality p-values: {p_values}, minimum: {min_p_value}")
            return min_p_value
        except Exception as e:
            logging.error(f"Granger causality test failed: {e}")
            return 1.0  # Non-significant on failure
