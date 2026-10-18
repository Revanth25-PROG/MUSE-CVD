"""
Wrapper to execute the master MUSE-CVD pipeline from the implementation folder.
"""
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from run_pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
