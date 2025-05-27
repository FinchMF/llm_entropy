import os
import time
import logging
from functools import wraps
from typing import Optional, Callable

logger = logging.getLogger('llm_entropy.validator')

def validate_save(filepath: str, timeout: int = 5) -> bool:
    """Validate that a file exists and is non-empty."""
    logger.info(f"Starting validation of file: {filepath}")
    start_time = time.time()
    
    # Check directory exists
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        logger.error(f"Directory does not exist: {dirpath}")
        return False
        
    while time.time() - start_time < timeout:
        if os.path.exists(filepath):
            try:
                size = os.path.getsize(filepath)
                if size > 0:
                    logger.info(f"✓ File validated successfully:")
                    logger.info(f"  - Path: {filepath}")
                    logger.info(f"  - Size: {size} bytes")
                    logger.info(f"  - Directory: {dirpath}")
                    return True
                logger.warning(f"⚠ File exists but is empty: {filepath}")
            except OSError as e:
                logger.error(f"Error checking file: {filepath}")
                logger.error(f"Details: {str(e)}")
        time.sleep(0.1)
        
    logger.error(f"✗ Validation failed for: {filepath}")
    logger.error(f"  - Timeout after {timeout} seconds")
    return False

def validate_file_operation(operation_name: str = "file operation"):
    """Decorator for validating file operations."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract filepath from args or kwargs
            filepath = kwargs.get('filepath', None)
            if filepath is None and len(args) > 0:
                filepath = args[-1]
            
            if not isinstance(filepath, str):
                raise ValueError("filepath must be a string")
            
            # Log pre-operation
            logger.info(f"Starting {operation_name}: {filepath}")
            
            # Execute operation
            result = func(*args, **kwargs)
            
            # Log post-operation
            logger.info(f"Completed {operation_name}: {filepath}")
            
            # Validate file
            if not validate_save(filepath):
                raise IOError(f"Failed to validate {operation_name}: {filepath}")
            
            return result
        return wrapper
    return decorator

class FileValidator:
    """Validator for file operations."""
    
    @staticmethod
    def validate_file_write(filepath: str, timeout: int = 5) -> bool:
        """Validate that a file exists and is non-empty."""
        return validate_save(filepath, timeout)

    @staticmethod
    def ensure_dir_exists(dirpath: str) -> None:
        """Ensure directory exists, create if it doesn't."""
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
            logger.info(f"Created directory: {dirpath}")
