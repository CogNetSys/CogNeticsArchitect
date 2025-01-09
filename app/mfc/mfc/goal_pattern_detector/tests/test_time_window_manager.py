# mfc/mfc/goal_pattern_detector/tests/test_time_window_manager.py

import unittest
from goal_pattern_detector.time_window_manager import TimeWindowManager

class TestTimeWindowManagerCorrelation(unittest.TestCase):
    def setUp(self):
        self.time_window_manager = TimeWindowManager(fixed_size=100, overlap_ratio=0.5, max_lag=5)

    def test_cross_correlation_positive(self):
        ca1 = [1, 2, 3, 4, 5]
        ca2 = [2, 4, 6, 8, 10]
        score = self.time_window_manager.calculate_cross_correlation(ca1, ca2)
        self.assertAlmostEqual(score, 1.0)

    def test_cross_correlation_negative(self):
        ca1 = [1, 2, 3, 4, 5]
        ca2 = [5, 4, 3, 2, 1]
        score = self.time_window_manager.calculate_cross_correlation(ca1, ca2)
        self.assertAlmostEqual(score, -1.0)

    def test_mutual_information(self):
        ca1 = [1, 2, 3, 4, 5]
        ca2 = [2, 4, 6, 8, 10]
        mi = self.time_window_manager.calculate_mutual_information(ca1, ca2, bins=5)
        self.assertGreater(mi, 0)

    def test_granger_causality(self):
        ca1 = [1, 2, 3, 4, 5, 6]
        ca2 = [2, 4, 6, 8, 10, 12]
        p_value = self.time_window_manager.calculate_granger_causality(ca1, ca2)
        self.assertLess(p_value, 0.05)

    def test_find_correlated_cas(self):
        window_data = {
            'CA1': [1, 2, 3, 4, 5],
            'CA2': [2, 4, 6, 8, 10],
            'CA3': [5, 4, 3, 2, 1]
        }
        correlated = self.time_window_manager.find_correlated_cas(
            window_data=window_data,
            methods=['cross_correlation', 'mutual_information', 'granger_causality'],
            threshold=0.8
        )
        self.assertIn(('CA1', 'CA2', 1.0), correlated)
        self.assertNotIn(('CA1', 'CA3', -1.0), correlated)  # Since threshold=0.8 and correlation=-1.0 < 0.8

if __name__ == '__main__':
    unittest.main()
