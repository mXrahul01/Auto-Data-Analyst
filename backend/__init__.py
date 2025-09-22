"""
Auto-Data-Analyst Backend Package

Provides enterprise-grade AI-powered data analysis services:
- Automated ML pipelines
- Feature engineering
- Real-time insights
- Secure, production-ready API
"""

__version__ = "2.0.0"
__author__ = "Rahul Talvar"
__email__ = "rahultalvar902@gmail.com"

# Initialize package-level logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Ensure environment variables are loaded early
from backend.config import settings, validate_and_setup_config
validate_and_setup_config()

# Expose key modules for convenience
from backend.main import app  # FastAPI app instance

__all__ = [
    "app",
    "settings",
    "validate_and_setup_config",
    "__version__",
    "__author__",
    "__email__",
]
