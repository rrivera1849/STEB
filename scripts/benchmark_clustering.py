"""Backward-compat shim. Equivalent to: python -m scripts.benchmark_clustering"""
import os
import runpy
import sys

# Add project root so the package import resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

runpy.run_module("scripts.benchmark_clustering", run_name="__main__", alter_sys=True)
