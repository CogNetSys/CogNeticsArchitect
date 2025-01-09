# mfc/mfc/goal_pattern_detector/tests/test_goal_pattern_detector.py

import unittest
from unittest.mock import MagicMock, patch
from goal_pattern_detector.goal_pattern_detector import GoalPatternDetector
from goal_pattern_detector.context_embedding import ContextEmbedding
from goal_pattern_detector.time_window_manager import TimeWindowManager

class TestGoalPatternDetector(unittest.TestCase):
    def setUp(self):
        self.context_embedding = ContextEmbedding()
        self.time_window_manager = TimeWindowManager()
        self.llm_api_key = "test-api-key"
        self.detector = GoalPatternDetector(
            context_embedding=self.context_embedding,
            time_window_manager=self.time_window_manager,
            llm_api_key=self.llm_api_key,
            significance_threshold=0.05,
            importance_scores={('CA1', 'CA2'): 2.0}
        )

    @patch('openai.ChatCompletion.create')
    def test_derive_rules_from_pattern_success(self, mock_create):
        mock_create.return_value = {
            'choices': [
                {'message': {'content': 'IF CA1 is active and CA2 exceeds threshold, THEN perform Action1.'}}
            ]
        }
        pattern = ('CA1', 'CA2')
        context = {'ca_states': {'CA1': [1,2,3], 'CA2': [2,4,6]}}
        frequency = 10
        significance = 0.01
        importance = 2.0
        rule = self.detector._derive_rules_from_pattern(pattern, context, frequency, significance, importance)
        expected_rule = 'IF CA1 is active and CA2 exceeds threshold, THEN perform Action1.'
        self.assertEqual(rule, expected_rule)

    @patch('openai.ChatCompletion.create')
    def test_derive_rules_from_pattern_failure(self, mock_create):
        mock_create.side_effect = Exception("API Error")
        pattern = ('CA1', 'CA2')
        context = {'ca_states': {'CA1': [1,2,3], 'CA2': [2,4,6]}}
        frequency = 10
        significance = 0.01
        importance = 2.0
        rule = self.detector._derive_rules_from_pattern(pattern, context, frequency, significance, importance)
        self.assertEqual(rule, "IF [conditions] THEN [actions].")

    def test_detect_patterns_basic(self):
        # Mock the LLM call to prevent actual API usage during testing
        with patch.object(self.detector, '_derive_rules_from_pattern', return_value="IF CA1 is active and CA2 exceeds threshold, THEN perform Action1."):
            data = {
                'CA1': [1, 2, 3, 4, 5],
                'CA2': [2, 4, 6, 8, 10],
                'CA3': [5, 4, 3, 2, 1]
            }
            patterns = self.detector.detect_patterns(data, adaptive=False, triggers=[])
            self.assertEqual(len(patterns), 1)
            self.assertEqual(patterns[0]['pattern'], ('CA1', 'CA2'))
            self.assertEqual(patterns[0]['rule'], "IF CA1 is active and CA2 exceeds threshold, THEN perform Action1.")

if __name__ == '__main__':
    unittest.main()
