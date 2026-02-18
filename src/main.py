"""
MAIN.PY (SHIM)
==============
This file is a wrapper for backward compatibility.
It redirects to the new modular entry point: src.cli

Recommended Usage:
    python -m src.cli [ARGS]
    OR
    make run
"""
import sys
import os

# Ensure the project root is in sys.path so 'src' is treated as a package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cli import main

if __name__ == "__main__":
    main()
