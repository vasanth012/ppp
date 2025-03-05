# test_librosa.py
"""
Title: Verify Librosa Installation
Description: This script checks if the librosa library is installed correctly
and prints its version.
"""

import librosa

try:
    print("Librosa is installed. Version:", librosa.__version__)
except ModuleNotFoundError:
    print("Librosa is not installed. Please install it using: pip install librosa")
