import os
import logging
from datetime import datetime
import uuid
from typing import Optional

def setup_logging(run_id: Optional[str] = None) -> str:
    """Configure logging for the current analysis run.
    
    Args:
        run_id: Optional identifier for the run. If None, generates a unique ID.
    
    Returns:
        str: The run ID used for this analysis session.
    """
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up file handler
    log_file = os.path.join(log_dir, f"analysis_run_{run_id}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Configure formatting
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Set up root logger
    root_logger = logging.getLogger('llm_entropy')
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return run_id
