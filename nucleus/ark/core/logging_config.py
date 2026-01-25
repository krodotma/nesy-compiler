#!/usr/bin/env python3
"""
logging_config.py - Centralized logging configuration for ARK

Sets up consistent logging across all ARK modules.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_ark_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for all ARK modules.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for log output
        format_string: Optional custom format string
    """
    if format_string is None:
        format_string = "[%(levelname)s] %(name)s: %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Configure all ARK loggers
    ark_modules = [
        "ARK",
        "ARK.Repository",
        "ARK.Gates",
        "ARK.Gates.Inertia",
        "ARK.Gates.Entelecheia",
        "ARK.Gates.Homeostasis",
        "ARK.Rhizom",
        "ARK.Rhizom.Etymology",
        "ARK.Rhizom.Lineage",
        "ARK.Portal",
        "ARK.Portal.Ingest",
        "ARK.Portal.Distill",
        "ARK.Synthesis",
        "ARK.Integration",
    ]
    
    for module in ark_modules:
        logger = logging.getLogger(module)
        logger.setLevel(level)
        logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)


# Module-level loggers for easy import
def get_logger(name: str) -> logging.Logger:
    """Get a logger for an ARK module."""
    full_name = f"ARK.{name}" if not name.startswith("ARK") else name
    return logging.getLogger(full_name)
