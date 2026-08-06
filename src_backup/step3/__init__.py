# src/step3/__init__.py

"""
Step 3: Projection Method Package.

This package implements the Projection Method for incompressible flow.
Compliance:
- Rule 8 (API Minimalism): Only the orchestrator is exposed.
- Rule 4 (SSoT): Internal modules are isolated to prevent redundant data access paths.
"""

from src.step3.orchestrate_step3 import orchestrate_step3

# We expose ONLY the orchestrator. The individual solver components (corrector, 
# ppe_solver, predictor) are kept internal to enforce the API Minimalism 
# mandate and prevent direct attribute access bypasses.
__all__ = ["orchestrate_step3"]