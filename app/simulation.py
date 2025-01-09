import os
import time
import random
import logging
import numpy as np
from collections import defaultdict
from mfc.mfc.goal_pattern_detector.context_embedding import ContextEmbedding
from mfc.mfc.goal_pattern_detector.time_window_manager import TimeWindowManager
from mfc.mfc.goal_pattern_detector.goal_pattern_detector import GoalPatternDetector
from mfc.agents.CA_agent import CAAgent
from mfc.mfc_manager import MFCManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Initialize components
    context_embedding = ContextEmbedding()
    time_window_manager = TimeWindowManager(fixed_size=10, overlap_ratio=0.5, max_lag=5)
    llm_api_key = os.getenv("OPENAI_API_KEY")  # Ensure this is set in your environment
    if llm_api_key is None:
        logging.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        exit(1)
    goal_pattern_detector = GoalPatternDetector(
        context_embedding=context_embedding,
        time_window_manager=time_window_manager,
        llm_api_key=llm_api_key,
        significance_threshold=0.05,
        importance_scores={('CA1', 'CA2'): 2.0}
    )

    # Initialize MFC Model with 3 CAs
    mfc_model = MFCManager(
        config={},  # Add appropriate configuration
        context_embedding=context_embedding,
        time_window_manager=time_window_manager,
        llm_api_key=llm_api_key
    )
    ca1 = CAAgent(unique_id='CA1', initial_state=0)
    ca2 = CAAgent(unique_id='CA2', initial_state=0)
    ca3 = CAAgent(unique_id='CA3', initial_state=0)
    mfc_model.add_agent(ca1)
    mfc_model.add_agent(ca2)
    mfc_model.add_agent(ca3)

    # Run Simulation for 20 steps
    mfc_model.run_mfc(steps=50)