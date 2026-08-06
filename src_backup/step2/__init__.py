# src/step2/__init__.py

"""
Step 2: Topology Assembly Layer
This module manages the instantiation of the topological graph (StencilBlocks).
Access is restricted to the orchestrator to maintain API Minimalism (Rule 8).
"""

from .orchestrate_step2 import orchestrate_step2

# Rule 8: API Minimalism. 
# Explicitly export only the orchestrator to prevent leakage of internal 
# wiring logic (Cell/StencilBlock assembly) to higher-level application code.
__all__ = ["orchestrate_step2"]