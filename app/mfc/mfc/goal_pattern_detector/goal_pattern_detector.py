# mfc/mfc/goal_pattern_detector/goal_pattern_detector.py

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx
from .context_embedding import ContextEmbedding
from .time_window_manager import TimeWindowManager
from datetime import datetime, timedelta
import openai
import os
import logging
import re
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GoalPatternDetector:
    def __init__(self,
                 significance_threshold: float = 0.05,
                 min_pattern_length: int = 3,
                 initial_threshold: float = 0.9,
                 min_threshold: float = 0.6,
                 adaptation_rate: float = 0.05,
                 fixed_window_size: int = 100,
                 max_windows: int = 100,
                 llm_api_key: str = None):
        """
        Initializes the GoalPatternDetector.

        Args:
            significance_threshold (float): Threshold for pattern significance (p-value).
            min_pattern_length (int): Minimum number of occurrences for a pattern to be considered.
            initial_threshold (float): Starting threshold for dynamic adjustment.
            min_threshold (float): Minimum threshold limit.
            adaptation_rate (float): Rate at which the threshold adapts based on performance.
            fixed_window_size (int): Size of fixed windows.
            max_windows (int): Maximum number of windows to keep in memory.
            llm_api_key (str): API key for the LLM (e.g., OpenAI).
        """
        self.significance_threshold = significance_threshold
        self.min_pattern_length = min_pattern_length
        self.current_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.adaptation_rate = adaptation_rate
        self.context_embedding = ContextEmbedding()
        self.time_window_manager = TimeWindowManager(fixed_size=fixed_window_size, max_lag=5)
        self.pattern_history = defaultdict(list)
        self.pattern_graph = nx.DiGraph()
        self.llm_api_key = llm_api_key or os.getenv("OPENAI_API_KEY")
        if self.llm_api_key is None:
            raise ValueError("LLM API key must be provided or set as an environment variable.")
        openai.api_key = self.llm_api_key
        self.importance_scores = {
            ('CA1', 'CA2'): 1.2,
            ('CA1', 'CA3'): 1.5,
            ('CA2', 'CA3'): 1.8
        }
        self.patterns = defaultdict(int)  # Initialize pattern frequency tracker
        self.rules = []  # List to store generated rules
        self.rule_cache = {}  # Cache for generated rules to avoid redundancy

        logging.info("Initialized GoalPatternDetector.")

    def adapt_threshold(self, system_performance: float):
        """Dynamically adjust significance threshold based on system performance."""
        if system_performance > 0.8:  # System performing well, can look for subtle patterns
            self.current_threshold = max(
                self.min_threshold,
                self.current_threshold - self.adaptation_rate
            )
            logging.info(f"System performance high. Decreased threshold to {self.current_threshold}.")
        else:  # System needs to focus on stronger patterns
            self.current_threshold = min(
                0.9,
                self.current_threshold + self.adaptation_rate
            )
            logging.info(f"System performance low. Increased threshold to {self.current_threshold}.")

    def record_state_transition(self, ca_id: str, old_state: dict, new_state: dict,
                                context: dict, conditions: List[dict] = None,
                                trigger_event: Optional[str] = None):
        """
        Records a state transition with context and optional conditions.

        Args:
            ca_id (str): The ID of the CA.
            old_state (dict): The previous state of the CA.
            new_state (dict): The new state of the CA.
            context (dict): Contextual information related to the transition.
            conditions (list): Optional list of conditions triggering the transition.
            trigger_event (str): Optional trigger event.
        """
        context_text = self.context_embedding._context_to_text(context)
        context_embedding = self.context_embedding.get_embedding(context_text)

        transition = {
            'old_state': old_state,
            'new_state': new_state,
            'context': context_text,
            'context_embedding': context_embedding,
            'conditions': conditions or [],
            'timestamp': datetime.now(),
            'trigger_event': trigger_event
        }
        self.pattern_history[ca_id].append(transition)
        self._update_pattern_graph(transition)

        logging.debug(f"Recorded transition for {ca_id}: {old_state} -> {new_state}")

    def detect_temporal_patterns(self, data: Dict[str, int], adaptive: bool = False, triggers: Optional[List[int]] = None) -> List[Dict]:
        """
        Detects patterns within the provided data.

        Args:
            data (dict): Dictionary of CA states where keys are CA IDs and values are current states.
            adaptive (bool): Whether to create adaptive windows based on triggers.
            triggers (list): List of trigger events to adjust windowing.

        Returns:
            list: List of detected patterns with their details.
        """
        # Prepare window data
        window_data = {ca_id: [agent_state] for ca_id, agent_state in data.items()}
        windows = self.time_window_manager.create_time_windows(window_data, adaptive=adaptive, triggers=triggers)
        detected_patterns = []

        for window in windows:
            prepared_data = self._prepare_window_data(window)
            correlated_cas = self.time_window_manager.find_correlated_cas(
                prepared_data,
                methods=['cross_correlation', 'mutual_information', 'granger_causality'],
                threshold=self.current_threshold
            )
            for ca1, ca2, score in correlated_cas:
                pattern = (ca1, ca2)
                self.patterns[pattern] += 1
                significance = self._calculate_significance(
                    pattern=pattern,
                    frequency=self.patterns[pattern],
                    total_sequences=sum(self.patterns.values()),
                    expected_frequency=1/len(self.patterns) if self.patterns else 0.01
                )
                if significance < self.significance_threshold:
                    importance = self.importance_scores.get(pattern, 1.0)
                    rule = self._derive_rules_from_pattern(
                        pattern=pattern,
                        context=window,  # Pass relevant context if available
                        frequency=self.patterns[pattern],
                        significance=significance,
                        importance=importance
                    )
                    detected_patterns.append({
                        'pattern': pattern,
                        'frequency': self.patterns[pattern],
                        'significance': significance,
                        'importance': importance,
                        'rule': rule
                    })
                    # Trigger adaptive window if necessary
                    if 'pattern_detected' in self.time_window_manager.trigger_events:
                        if triggers is not None:
                            current_step = len(next(iter(window.values()))) - self.time_window_manager.step_size
                            triggers.append(current_step)  # Example trigger based on pattern detection

        logging.info(f"Detected {len(detected_patterns)} patterns in current window.")
        return detected_patterns

    def _prepare_window_data(self, window: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
        """
        Prepares window data for correlation analysis.

        Args:
            window (dict): Window data with CA IDs as keys and state sequences as values.

        Returns:
            dict: Prepared window data with NumPy arrays.
        """
        prepared_data = {}
        for ca_id, states in window.items():
            prepared_data[ca_id] = np.array(states)
        return prepared_data

    def _update_pattern_graph(self, transition: dict):
        """
        Updates the pattern graph with new transition information.

        Args:
            transition (dict): The transition data.
        """
        old_state_key = self._state_to_key(transition['old_state'])
        new_state_key = self._state_to_key(transition['new_state'])

        # Add nodes
        self.pattern_graph.add_node(old_state_key)
        self.pattern_graph.add_node(new_state_key)

        # Add or update edge
        if self.pattern_graph.has_edge(old_state_key, new_state_key):
            self.pattern_graph[old_state_key][new_state_key]['weight'] += 1
        else:
            self.pattern_graph.add_edge(old_state_key, new_state_key, weight=1)

        # Optionally, store context embeddings
        if 'contexts' not in self.pattern_graph[old_state_key][new_state_key]:
            self.pattern_graph[old_state_key][new_state_key]['contexts'] = []
        self.pattern_graph[old_state_key][new_state_key]['contexts'].append(transition['context_embedding'])

        logging.debug(f"Updated pattern graph: {old_state_key} -> {new_state_key}")

    def _state_to_key(self, state: dict) -> str:
        """
        Converts a state dictionary to a string key for graph nodes.

        Args:
            state (dict): The state dictionary.

        Returns:
            str: String representation of the state.
        """
        return str(sorted(state.items()))

    def _calculate_significance(self, pattern: Tuple[str, str], frequency: int, total_sequences: int, expected_frequency: float, method: str = 'chi_squared') -> float:
        """
        Calculates the statistical significance of a pattern.

        Args:
            pattern (tuple): The detected pattern.
            frequency (int): Observed frequency of the pattern.
            total_sequences (int): Total number of sequences.
            expected_frequency (float): Expected frequency under the null hypothesis.
            method (str): Statistical test to use ('chi_squared', 'ks_test').

        Returns:
            float: Adjusted p-value indicating significance.
        """
        if method == 'chi_squared':
            # Avoid division by zero or invalid values
            observed = [frequency, max(total_sequences - frequency, 1)]
            expected = [expected_frequency * total_sequences, max(total_sequences * (1 - expected_frequency), 1)]
            chi2, p = stats.chisquare(f_obs=observed, f_exp=expected)
        elif method == 'ks_test':
            observed = [frequency, max(total_sequences - frequency, 1)]
            expected = [expected_frequency * total_sequences, max(total_sequences * (1 - expected_frequency), 1)]
            ks_stat, p = stats.ks_2samp(observed, expected)
        else:
            raise ValueError(f"Unsupported statistical method: {method}")

        # Adjust p-value based on importance score
        importance = self.importance_scores.get(pattern, 1.0)
        if importance <= 0:
            adjusted_p = p  # No adjustment if importance is non-positive
        else:
            adjusted_p = p / importance  # Example adjustment: higher importance reduces p-value threshold
        adjusted_p = min(adjusted_p, 1.0)  # Ensure p-value does not exceed 1.0

        logging.debug(f"Calculated significance for pattern {pattern}: p-value={adjusted_p}")
        return adjusted_p

    def _derive_rules_from_pattern(self, pattern: Tuple[str, str], context: dict, frequency: int, significance: float, importance: float) -> str:
        """
        Generates a descriptive IF-THEN rule based on the detected pattern using GPT-4.

        Args:
            pattern (tuple): The detected pattern.
            context (dict): Context elements related to the pattern.
            frequency (int): Frequency of the pattern.
            significance (float): Statistical significance of the pattern.
            importance (float): Domain-specific importance score.

        Returns:
            str: Generated rule description.
        """
        # Check if rule already exists in cache
        pattern_key = f"{pattern[0]}_{pattern[1]}"
        if pattern_key in self.rule_cache:
            logging.debug(f"Rule for pattern {pattern} retrieved from cache.")
            return self.rule_cache[pattern_key]

        prompt = f"""
Generate a descriptive IF-THEN rule based on the following pattern:

Pattern Details:
- CAs Involved: {pattern[0]}, {pattern[1]}
- Frequency: {frequency}
- Significance: {significance}
- Importance: {importance}

Available Actions:
- Action1: Increase resource allocation by 10%.
- Action2: Trigger an alert to notify the system administrator.

Conditions:
- Condition1: {pattern[0]}'s state is increasing steadily.
- Condition2: {pattern[1]}'s state exceeds the threshold of 5.

Constraints:
- Maximum rule length: 2 sentences
- Rule format: IF [conditions] THEN [actions]

Example Rule 1:
IF {pattern[0]}'s state is below 2 and {pattern[1]}'s state is above 5, THEN perform Action1.

Example Rule 2:
IF {pattern[0]}'s state is stable and {pattern[1]}'s state is increasing, THEN perform Action2.

Generated Rule:
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in generating IF-THEN rules based on detected patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.5,
            )
            rule = response.choices[0].message['content'].strip()

            # Calculate composite score
            composite_score = self.evaluate_rule_composite_score(rule)

            # Define a threshold for accepting rules
            if composite_score >= 0.8:
                # Check for redundancy
                if not self.is_rule_redundant(rule, self.rules):
                    self.rules.append(rule)
                    self.rule_cache[pattern_key] = rule  # Cache the valid rule
                    logging.info(f"Generated and validated new rule: {rule}")
                    return rule
                else:
                    logging.info(f"Generated rule is redundant: {rule}")
            else:
                logging.warning(f"Generated rule failed composite evaluation: {rule} (Score: {composite_score})")

            return "IF [conditions] THEN [actions]."  # Default fallback rule
        except Exception as e:
            logging.error(f"Error generating rule with LLM: {e}")
            return "IF [conditions] THEN [actions]."  # Default fallback rule

    def evaluate_rule_composite_score(self, rule: str) -> float:
        """
        Calculates a composite score for the rule based on multiple evaluation metrics.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            float: Composite score between 0 and 1.
        """
        scores = self.evaluate_rule(rule)
        semantic_coherence = self.evaluate_semantic_coherence(rule)
        diversity_score = self.evaluate_rule_diversity(rule, self.rules)
        structure_valid = self.evaluate_rule_structure(rule)
        actionability = self.evaluate_actionability(rule)

        # Define weights for each metric
        weights = {
            'length': 0.2,
            'complexity': 0.2,
            'keywords': 0.2,
            'actionability': 0.2,
            'structure': 0.2,
            'semantic_coherence': 0.1,
            'diversity_score': 0.1
        }

        # Calculate weighted score
        composite_score = (
            weights['length'] * scores['length'] +
            weights['complexity'] * scores['complexity'] +
            weights['keywords'] * scores['keywords'] +
            weights['actionability'] * scores['actionability'] +
            weights['structure'] * scores['structure'] +
            weights['semantic_coherence'] * (semantic_coherence > 0.5) +  # Binary score
            weights['diversity_score'] * (diversity_score > 0.3)        # Binary score
        )

        logging.debug(f"Composite score for rule: {composite_score}")
        return composite_score

    def evaluate_rule(self, rule: str) -> Dict[str, bool]:
        """
        Evaluates the quality of a generated rule based on multiple metrics.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            dict: Evaluation scores for each metric.
        """
        scores = {
            'length': self.evaluate_rule_length(rule),
            'complexity': self.evaluate_rule_complexity(rule),
            'keywords': self.evaluate_rule_keywords(rule),
            'actionability': self.evaluate_actionability(rule),
            'structure': self.evaluate_rule_structure(rule)
        }
        return scores

    def evaluate_rule_length(self, rule: str, max_words: int = 20) -> bool:
        """
        Evaluates if the rule length is within the specified limit.

        Args:
            rule (str): The generated IF-THEN rule.
            max_words (int): Maximum allowed words.

        Returns:
            bool: True if within limit, else False.
        """
        word_count = len(rule.split())
        return word_count <= max_words

    def evaluate_rule_complexity(self, rule: str, max_conditions: int = 2, max_actions: int = 2) -> bool:
        """
        Evaluates if the rule complexity is within specified limits.

        Args:
            rule (str): The generated IF-THEN rule.
            max_conditions (int): Maximum allowed conditions.
            max_actions (int): Maximum allowed actions.

        Returns:
            bool: True if within limits, else False.
        """
        try:
            conditions = rule.lower().split("if")[1].split("then")[0].strip().split(" and ")
            actions = rule.lower().split("then")[1].strip().split(" and ")
            return len(conditions) <= max_conditions and len(actions) <= max_actions
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return False

    def evaluate_rule_keywords(self, rule: str, required_keywords: Optional[Dict[str, List[str]]] = None) -> bool:
        """
        Evaluates if the rule contains required keywords in conditions and actions.

        Args:
            rule (str): The generated IF-THEN rule.
            required_keywords (dict): Dictionary with 'conditions' and 'actions' as keys and list of keywords as values.

        Returns:
            bool: True if all required keywords are present, else False.
        """
        if required_keywords is None:
            required_keywords = {
                'conditions': ['state', 'threshold', 'steady', 'increasing'],
                'actions': ['increase', 'trigger', 'notify', 'allocate']
            }

        try:
            conditions = rule.lower().split("if")[1].split("then")[0]
            actions = rule.lower().split("then")[1]
            condition_check = all(keyword in conditions for keyword in required_keywords['conditions'])
            action_check = all(keyword in actions for keyword in required_keywords['actions'])
            return condition_check and action_check
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return False

    def evaluate_actionability(self, rule: str) -> bool:
        """
        Evaluates if the rule actions are actionable.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            bool: True if actionable, else False.
        """
        actionable_verbs = ['increase', 'decrease', 'allocate', 'trigger', 'notify', 'adjust', 'reset']
        try:
            actions = rule.lower().split("then")[1]
            return any(verb in actions for verb in actionable_verbs)
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return False

    def evaluate_rule_structure(self, rule: str) -> bool:
        """
        Evaluates whether the rule follows the IF-THEN structure.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            bool: True if the rule follows the IF-THEN structure, else False.
        """
        pattern = r'^IF\s+.+\s+THEN\s+.+\.$'
        return re.match(pattern, rule.strip(), re.IGNORECASE) is not None

    def evaluate_semantic_coherence(self, rule: str) -> float:
        """
        Evaluates the semantic coherence between conditions and actions in a rule.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            float: Semantic coherence score (0 to 1).
        """
        try:
            conditions = rule.split("IF")[1].split("THEN")[0].strip()
            actions = rule.split("THEN")[1].strip().strip('.')
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return 0.0

        conditions_embedding = self.context_embedding.get_embedding(conditions)
        actions_embedding = self.context_embedding.get_embedding(actions)

        if conditions_embedding.size == 0 or actions_embedding.size == 0:
            logging.warning("One or both embeddings are empty for semantic coherence.")
            return 0.0

        similarity = self.context_embedding.calculate_similarity(conditions_embedding, actions_embedding)
        logging.debug(f"Semantic coherence between conditions and actions: {similarity}")
        return similarity

    def evaluate_rule_diversity(self, new_rule: str, existing_rules: List[str]) -> float:
        """
        Evaluates the diversity of a new rule compared to existing rules.

        Args:
            new_rule (str): The newly generated rule.
            existing_rules (List[str]): List of existing rules.

        Returns:
            float: Diversity score (0 to 1), where 1 is highly diverse.
        """
        new_embedding = self.context_embedding.get_embedding(new_rule)
        similarities = [self.context_embedding.calculate_similarity(new_embedding, self.context_embedding.get_embedding(rule)) for rule in existing_rules]
        max_similarity = max(similarities) if similarities else 0
        diversity_score = 1 - max_similarity
        logging.debug(f"Diversity score for new rule: {diversity_score}")
        return diversity_score

    def is_rule_redundant(self, new_rule: str, existing_rules: List[str], similarity_threshold: float = 0.9) -> bool:
        """
        Determines if a new rule is redundant based on similarity to existing rules.

        Args:
            new_rule (str): The newly generated rule.
            existing_rules (List[str]): List of existing rules.
            similarity_threshold (float): Threshold above which rules are considered redundant.

        Returns:
            bool: True if the rule is redundant, False otherwise.
        """
        new_embedding = self.context_embedding.get_embedding(new_rule)
        for rule in existing_rules:
            existing_embedding = self.context_embedding.get_embedding(rule)
            similarity = self.context_embedding.calculate_similarity(new_embedding, existing_embedding)
            if similarity >= similarity_threshold:
                logging.info(f"New rule is redundant with an existing rule. Similarity: {similarity}")
                return True
        return False

    def _derive_rules_from_pattern(self, pattern: Tuple[str, str], context: dict, frequency: int, significance: float, importance: float) -> str:
        """
        Generates a descriptive IF-THEN rule based on the detected pattern using GPT-4.

        Args:
            pattern (tuple): The detected pattern.
            context (dict): Context elements related to the pattern.
            frequency (int): Frequency of the pattern.
            significance (float): Statistical significance of the pattern.
            importance (float): Domain-specific importance score.

        Returns:
            str: Generated rule description or a default rule if validation fails.
        """
        # Check if rule already exists in cache
        pattern_key = f"{pattern[0]}_{pattern[1]}"
        if pattern_key in self.rule_cache:
            logging.debug(f"Rule for pattern {pattern} retrieved from cache.")
            return self.rule_cache[pattern_key]

        prompt = f"""
Generate a descriptive IF-THEN rule based on the following pattern:

Pattern Details:
- CAs Involved: {pattern[0]}, {pattern[1]}
- Frequency: {frequency}
- Significance: {significance}
- Importance: {importance}

Available Actions:
- Action1: Increase resource allocation by 10%.
- Action2: Trigger an alert to notify the system administrator.

Conditions:
- Condition1: {pattern[0]}'s state is increasing steadily.
- Condition2: {pattern[1]}'s state exceeds the threshold of 5.

Constraints:
- Maximum rule length: 2 sentences
- Rule format: IF [conditions] THEN [actions]

Example Rule 1:
IF {pattern[0]}'s state is below 2 and {pattern[1]}'s state is above 5, THEN perform Action1.

Example Rule 2:
IF {pattern[0]}'s state is stable and {pattern[1]}'s state is increasing, THEN perform Action2.

Generated Rule:
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in generating IF-THEN rules based on detected patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.5,
            )
            rule = response.choices[0].message['content'].strip()

            # Calculate composite score
            composite_score = self.evaluate_rule_composite_score(rule)

            # Define a threshold for accepting rules
            if composite_score >= 0.8:
                # Check for redundancy
                if not self.is_rule_redundant(rule, self.rules):
                    self.rules.append(rule)
                    self.rule_cache[pattern_key] = rule  # Cache the valid rule
                    logging.info(f"Generated and validated new rule: {rule}")
                    return rule
                else:
                    logging.info(f"Generated rule is redundant: {rule}")
            else:
                logging.warning(f"Generated rule failed composite evaluation: {rule} (Score: {composite_score})")

            return "IF [conditions] THEN [actions]."  # Default fallback rule
        except Exception as e:
            logging.error(f"Error generating rule with LLM: {e}")
            return "IF [conditions] THEN [actions]."  # Default fallback rule

    def detect_complex_patterns(self):
        """
        Detects complex patterns including conditional, branching, and composite patterns.

        Returns:
            list: List of complex patterns detected.
        """
        patterns = []

        # Detect conditional patterns
        conditional_patterns = self._detect_conditional_patterns()
        patterns.extend(conditional_patterns)

        # Detect branching patterns
        branching_patterns = self._detect_branching_patterns()
        patterns.extend(branching_patterns)

        # Detect composite patterns
        composite_patterns = self._detect_composite_patterns()
        patterns.extend(composite_patterns)

        logging.info(f"Detected {len(patterns)} complex patterns.")
        return patterns

    def _detect_conditional_patterns(self) -> List[dict]:
        """
        Identifies patterns that occur under specific conditions.

        Returns:
            list: List of conditional patterns detected.
        """
        conditional_patterns = []

        for ca_id, transitions in self.pattern_history.items():
            condition_groups = defaultdict(list)

            # Group transitions by conditions
            for transition in transitions:
                condition_key = self._hash_conditions(transition['conditions'])
                condition_groups[condition_key].append(transition)

            # Analyze each condition group
            for conditions, group in condition_groups.items():
                if len(group) >= self.min_pattern_length:
                    pattern = self._analyze_conditional_group(group, conditions)
                    if pattern:
                        conditional_patterns.append(pattern)

        logging.debug(f"Detected {len(conditional_patterns)} conditional patterns.")
        return conditional_patterns

    def _detect_branching_patterns(self) -> List[dict]:
        """
        Identifies patterns with multiple possible outcome states.

        Returns:
            list: List of branching patterns detected.
        """
        branching_patterns = []

        # Identify nodes with multiple outgoing edges
        for node in self.pattern_graph.nodes():
            successors = list(self.pattern_graph.successors(node))
            if len(successors) > 1:
                transition_weights = {succ: self.pattern_graph[node][succ]['weight'] for succ in successors}
                transition_probabilities = {succ: weight / sum(transition_weights.values()) for succ, weight in transition_weights.items()}
                pattern = {
                    'type': 'branching',
                    'initial_state': node,
                    'possible_outcomes': successors,
                    'transition_probabilities': transition_probabilities
                }
                branching_patterns.append(pattern)

        logging.debug(f"Detected {len(branching_patterns)} branching patterns.")
        return branching_patterns

    def _detect_composite_patterns(self) -> List[dict]:
        """
        Identifies patterns involving multiple CAs.

        Returns:
            list: List of composite patterns detected.
        """
        composite_patterns = []

        # Placeholder for composite pattern detection logic
        # Example: Sequential correlations between multiple CAs

        logging.debug("Composite pattern detection not yet implemented.")
        return composite_patterns  # Currently empty

    def _analyze_conditional_group(self, group: List[dict], conditions: str) -> Optional[dict]:
        """
        Analyzes transitions that occur under specific conditions.

        Args:
            group (List[dict]): List of transitions under the same conditions.
            conditions (str): String representation of the conditions.

        Returns:
            Optional[dict]: Detected pattern details if significant, else None.
        """
        frequency = len(group)
        expected_frequency = 1 / len(self.patterns) if self.patterns else 0.01  # Avoid division by zero
        significance = self._calculate_significance(
            pattern=tuple(conditions.split('_')),
            frequency=frequency,
            total_sequences=sum(self.patterns.values()),
            expected_frequency=expected_frequency
        )
        if significance < self.significance_threshold:
            importance = self.importance_scores.get(tuple(conditions.split('_')), 1.0)  # Adjust as per conditions format
            rule = self._derive_rules_from_pattern(
                pattern=tuple(conditions.split('_')),  # Adjust as per conditions format
                context={},  # Provide relevant context if available
                frequency=frequency,
                significance=significance,
                importance=importance
            )
            return {
                'pattern': tuple(conditions.split('_')),  # Adjust as per conditions format
                'frequency': frequency,
                'significance': significance,
                'importance': importance,
                'rule': rule
            }
        return None

    def _hash_conditions(self, conditions: List[dict]) -> str:
        """
        Creates a hash key for a set of conditions.

        Args:
            conditions (list): List of condition dictionaries.

        Returns:
            str: String hash of the conditions.
        """
        return '_'.join([str(cond) for cond in conditions])

    def evaluate_rule_composite_score(self, rule: str) -> float:
        """
        Calculates a composite score for the rule based on multiple evaluation metrics.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            float: Composite score between 0 and 1.
        """
        scores = self.evaluate_rule(rule)
        semantic_coherence = self.evaluate_semantic_coherence(rule)
        diversity_score = self.evaluate_rule_diversity(rule, self.rules)
        structure_valid = self.evaluate_rule_structure(rule)
        actionability = self.evaluate_actionability(rule)

        # Define weights for each metric
        weights = {
            'length': 0.2,
            'complexity': 0.2,
            'keywords': 0.2,
            'actionability': 0.2,
            'structure': 0.2,
            'semantic_coherence': 0.1,
            'diversity_score': 0.1
        }

        # Calculate weighted score
        composite_score = (
            weights['length'] * scores['length'] +
            weights['complexity'] * scores['complexity'] +
            weights['keywords'] * scores['keywords'] +
            weights['actionability'] * scores['actionability'] +
            weights['structure'] * scores['structure'] +
            weights['semantic_coherence'] * (semantic_coherence > 0.5) +  # Binary score
            weights['diversity_score'] * (diversity_score > 0.3)        # Binary score
        )

        logging.debug(f"Composite score for rule: {composite_score}")
        return composite_score

    def evaluate_rule(self, rule: str) -> Dict[str, bool]:
        """
        Evaluates the quality of a generated rule based on multiple metrics.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            dict: Evaluation scores for each metric.
        """
        scores = {
            'length': self.evaluate_rule_length(rule),
            'complexity': self.evaluate_rule_complexity(rule),
            'keywords': self.evaluate_rule_keywords(rule),
            'actionability': self.evaluate_actionability(rule),
            'structure': self.evaluate_rule_structure(rule)
        }
        return scores

    def evaluate_rule_length(self, rule: str, max_words: int = 20) -> bool:
        """
        Evaluates if the rule length is within the specified limit.

        Args:
            rule (str): The generated IF-THEN rule.
            max_words (int): Maximum allowed words.

        Returns:
            bool: True if within limit, else False.
        """
        word_count = len(rule.split())
        return word_count <= max_words

    def evaluate_rule_complexity(self, rule: str, max_conditions: int = 2, max_actions: int = 2) -> bool:
        """
        Evaluates if the rule complexity is within specified limits.

        Args:
            rule (str): The generated IF-THEN rule.
            max_conditions (int): Maximum allowed conditions.
            max_actions (int): Maximum allowed actions.

        Returns:
            bool: True if within limits, else False.
        """
        try:
            conditions = rule.lower().split("if")[1].split("then")[0].strip().split(" and ")
            actions = rule.lower().split("then")[1].strip().split(" and ")
            return len(conditions) <= max_conditions and len(actions) <= max_actions
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return False

    def evaluate_rule_keywords(self, rule: str, required_keywords: Optional[Dict[str, List[str]]] = None) -> bool:
        """
        Evaluates if the rule contains required keywords in conditions and actions.

        Args:
            rule (str): The generated IF-THEN rule.
            required_keywords (dict): Dictionary with 'conditions' and 'actions' as keys and list of keywords as values.

        Returns:
            bool: True if all required keywords are present, else False.
        """
        if required_keywords is None:
            required_keywords = {
                'conditions': ['state', 'threshold', 'steady', 'increasing'],
                'actions': ['increase', 'trigger', 'notify', 'allocate']
            }

        try:
            conditions = rule.lower().split("if")[1].split("then")[0]
            actions = rule.lower().split("then")[1]
            condition_check = all(keyword in conditions for keyword in required_keywords['conditions'])
            action_check = all(keyword in actions for keyword in required_keywords['actions'])
            return condition_check and action_check
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return False

    def evaluate_actionability(self, rule: str) -> bool:
        """
        Evaluates if the rule actions are actionable.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            bool: True if actionable, else False.
        """
        actionable_verbs = ['increase', 'decrease', 'allocate', 'trigger', 'notify', 'adjust', 'reset']
        try:
            actions = rule.lower().split("then")[1]
            return any(verb in actions for verb in actionable_verbs)
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return False

    def evaluate_rule_structure(self, rule: str) -> bool:
        """
        Evaluates whether the rule follows the IF-THEN structure.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            bool: True if the rule follows the IF-THEN structure, else False.
        """
        pattern = r'^IF\s+.+\s+THEN\s+.+\.$'
        return re.match(pattern, rule.strip(), re.IGNORECASE) is not None

    def evaluate_semantic_coherence(self, rule: str) -> float:
        """
        Evaluates the semantic coherence between conditions and actions in a rule.

        Args:
            rule (str): The generated IF-THEN rule.

        Returns:
            float: Semantic coherence score (0 to 1).
        """
        try:
            conditions = rule.split("IF")[1].split("THEN")[0].strip()
            actions = rule.split("THEN")[1].strip().strip('.')
        except IndexError:
            logging.warning("Rule does not follow IF-THEN structure.")
            return 0.0

        conditions_embedding = self.context_embedding.get_embedding(conditions)
        actions_embedding = self.context_embedding.get_embedding(actions)

        if conditions_embedding.size == 0 or actions_embedding.size == 0:
            logging.warning("One or both embeddings are empty for semantic coherence.")
            return 0.0

        similarity = self.context_embedding.calculate_similarity(conditions_embedding, actions_embedding)
        logging.debug(f"Semantic coherence between conditions and actions: {similarity}")
        return similarity

    def evaluate_rule_diversity(self, new_rule: str, existing_rules: List[str]) -> float:
        """
        Evaluates the diversity of a new rule compared to existing rules.

        Args:
            new_rule (str): The newly generated rule.
            existing_rules (List[str]): List of existing rules.

        Returns:
            float: Diversity score (0 to 1), where 1 is highly diverse.
        """
        new_embedding = self.context_embedding.get_embedding(new_rule)
        similarities = [self.context_embedding.calculate_similarity(new_embedding, self.context_embedding.get_embedding(rule)) for rule in existing_rules]
        max_similarity = max(similarities) if similarities else 0
        diversity_score = 1 - max_similarity
        logging.debug(f"Diversity score for new rule: {diversity_score}")
        return diversity_score

    def is_rule_redundant(self, new_rule: str, existing_rules: List[str], similarity_threshold: float = 0.9) -> bool:
        """
        Determines if a new rule is redundant based on similarity to existing rules.

        Args:
            new_rule (str): The newly generated rule.
            existing_rules (List[str]): List of existing rules.
            similarity_threshold (float): Threshold above which rules are considered redundant.

        Returns:
            bool: True if the rule is redundant, False otherwise.
        """
        new_embedding = self.context_embedding.get_embedding(new_rule)
        for rule in existing_rules:
            existing_embedding = self.context_embedding.get_embedding(rule)
            similarity = self.context_embedding.calculate_similarity(new_embedding, existing_embedding)
            if similarity >= similarity_threshold:
                logging.info(f"New rule is redundant with an existing rule. Similarity: {similarity}")
                return True
        return False

    def _derive_rules_from_pattern(self, pattern: Tuple[str, str], context: dict, frequency: int, significance: float, importance: float) -> str:
        """
        Generates and validates a descriptive IF-THEN rule based on the detected pattern using GPT-4.

        Args:
            pattern (tuple): The detected pattern.
            context (dict): Context elements related to the pattern.
            frequency (int): Frequency of the pattern.
            significance (float): Statistical significance of the pattern.
            importance (float): Domain-specific importance score.

        Returns:
            str: Validated rule description or a default rule if validation fails.
        """
        # Check if rule already exists in cache
        pattern_key = f"{pattern[0]}_{pattern[1]}"
        if pattern_key in self.rule_cache:
            logging.debug(f"Rule for pattern {pattern} retrieved from cache.")
            return self.rule_cache[pattern_key]

        prompt = f"""
Generate a descriptive IF-THEN rule based on the following pattern:

Pattern Details:
- CAs Involved: {pattern[0]}, {pattern[1]}
- Frequency: {frequency}
- Significance: {significance}
- Importance: {importance}

Available Actions:
- Action1: Increase resource allocation by 10%.
- Action2: Trigger an alert to notify the system administrator.

Conditions:
- Condition1: {pattern[0]}'s state is increasing steadily.
- Condition2: {pattern[1]}'s state exceeds the threshold of 5.

Constraints:
- Maximum rule length: 2 sentences
- Rule format: IF [conditions] THEN [actions]

Example Rule 1:
IF {pattern[0]}'s state is below 2 and {pattern[1]}'s state is above 5, THEN perform Action1.

Example Rule 2:
IF {pattern[0]}'s state is stable and {pattern[1]}'s state is increasing, THEN perform Action2.

Generated Rule:
"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert in generating IF-THEN rules based on detected patterns."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.5,
            )
            rule = response.choices[0].message['content'].strip()

            # Calculate composite score
            composite_score = self.evaluate_rule_composite_score(rule)

            # Define a threshold for accepting rules
            if composite_score >= 0.8:
                # Check for redundancy
                if not self.is_rule_redundant(rule, self.rules):
                    self.rules.append(rule)
                    self.rule_cache[pattern_key] = rule  # Cache the valid rule
                    logging.info(f"Generated and validated new rule: {rule}")
                    return rule
                else:
                    logging.info(f"Generated rule is redundant: {rule}")
            else:
                logging.warning(f"Generated rule failed composite evaluation: {rule} (Score: {composite_score})")

            return "IF [conditions] THEN [actions]."  # Default fallback rule
        except Exception as e:
            logging.error(f"Error generating rule with LLM: {e}")
            return "IF [conditions] THEN [actions]."  # Default fallback rule

    def _detect_branching_patterns(self) -> List[dict]:
        """
        Identifies patterns with multiple possible outcome states.

        Returns:
            list: List of branching patterns detected.
        """
        branching_patterns = []

        # Identify nodes with multiple outgoing edges
        for node in self.pattern_graph.nodes():
            successors = list(self.pattern_graph.successors(node))
            if len(successors) > 1:
                transition_weights = {succ: self.pattern_graph[node][succ]['weight'] for succ in successors}
                transition_probabilities = {succ: weight / sum(transition_weights.values()) for succ, weight in transition_weights.items()}
                pattern = {
                    'type': 'branching',
                    'initial_state': node,
                    'possible_outcomes': successors,
                    'transition_probabilities': transition_probabilities
                }
                branching_patterns.append(pattern)

        logging.debug(f"Detected {len(branching_patterns)} branching patterns.")
        return branching_patterns

    def _detect_composite_patterns(self) -> List[dict]:
        """
        Identifies patterns involving multiple CAs.

        Returns:
            list: List of composite patterns detected.
        """
        composite_patterns = []

        # Placeholder for composite pattern detection logic
        # Example: Sequential correlations between multiple CAs

        logging.debug("Composite pattern detection not yet implemented.")
        return composite_patterns  # Currently empty

    def export_pattern_graph(self, filename: str = "pattern_graph.gexf"):
        """
        Exports the pattern graph to a GEXF file for visualization in tools like Gephi.

        Args:
            filename (str): The filename for the exported graph.
        """
        nx.write_gexf(self.pattern_graph, filename)
        logging.info(f"Exported pattern graph to {filename}")

    def prune_pattern_graph(self, min_weight: int = 5):
        """
        Prunes the pattern graph by removing edges with weight below the specified minimum.

        Args:
            min_weight (int): Minimum weight threshold for retaining edges.
        """
        edges_to_remove = [(u, v) for u, v, d in self.pattern_graph.edges(data=True) if d.get('weight', 0) < min_weight]
        self.pattern_graph.remove_edges_from(edges_to_remove)
        logging.info(f"Pruned {len(edges_to_remove)} edges from the pattern graph.")

    def load_pattern_graph(self, filename: str = "pattern_graph.gexf"):
        """
        Loads the pattern graph from a GEXF file.

        Args:
            filename (str): The filename of the graph to load.
        """
        self.pattern_graph = nx.read_gexf(filename)
        logging.info(f"Loaded pattern graph from {filename}")
