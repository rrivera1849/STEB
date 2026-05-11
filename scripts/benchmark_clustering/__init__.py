"""Benchmark clustering analysis for STEB.

Discovers which datasets within a task type measure similar constructs
(i.e., rank models the same way) using pairwise Spearman rank correlations
and hierarchical clustering, following the methodology from OLMo 3 Section 3.3.1.

Run via:
    python -m scripts.benchmark_clustering [args]
"""
