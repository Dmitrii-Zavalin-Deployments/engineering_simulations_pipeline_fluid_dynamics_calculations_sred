# src/step1/__init__.py

"""
Step 1: Orchestration Layer
This module serves as the primary entry point for the simulation assembly.
Access is restricted to the orchestrator to maintain API Minimalism (Rule 8).
"""

from .orchestrate_step1 import orchestrate_step1

# Explicitly export only the orchestrator to prevent 
# leakage of internal implementation details.
__all__ = ["orchestrate_step1"]