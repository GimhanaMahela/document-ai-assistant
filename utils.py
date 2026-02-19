"""
Utility functions for the Document AI Assistant.
Contains helper functions for file handling, text processing, and logging.
"""

import os
import logging
from datetime import datetime
from typing import List, Optional
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['data/uploaded', 'data/chroma_db', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")

def generate_file_hash(file_content: bytes) -> str:
    """
    Generate unique hash for file content.
    
    Args:
        file_content: Binary content of the file
    
    Returns:
        SHA-256 hash of the content
    """
    return hashlib.sha256(file_content).hexdigest()

def validate_file_type(file_name: str) -> bool:
    """
    Check if file type is supported.
    
    Args:
        file_name: Name of the uploaded file
    
    Returns:
        True if file type is supported, False otherwise
    """
    supported_extensions = ['.pdf', '.txt', '.md']
    ext = os.path.splitext(file_name)[1].lower()
    return ext in supported_extensions

def format_timestamp() -> str:
    """Return formatted current timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")