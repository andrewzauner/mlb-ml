import os
import sys

# Modules in this project use flat imports (`import data_loader`, etc.)
# rather than a package-relative layout, so tests need the repo root on
# sys.path regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
