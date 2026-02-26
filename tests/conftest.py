"""
Ensure the project root is on sys.path so all imports resolve.
Suppress noisy warnings during test collection.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# suppress third-party warnings that aren't our code
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)
