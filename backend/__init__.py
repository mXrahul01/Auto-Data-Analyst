
"""
Auto-Analyst Backend Package

Main backend package for the Auto-Analyst AI-powered data analysis platform.
Provides enterprise-grade ML capabilities with zero-code interface.
"""

__version__ = "2.0.0"
__author__ = "Auto-Analyst Team"
__email__ = "contact@auto-analyst.com"

# Package imports with error handling
import logging
import warnings

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

# Package metadata
__all__ = [
    "__version__",
    "__author__", 
    "__email__"
]

