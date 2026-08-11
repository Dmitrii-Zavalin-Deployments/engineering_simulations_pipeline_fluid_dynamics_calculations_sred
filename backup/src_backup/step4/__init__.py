# src/step4/__init__.py

"""
Step 5: Archivist Package.

This package manages the serialization of the simulation state to disk.
Compliance:
- Rule 8 (API Minimalism): Exposes only the primary orchestration entry point.
- Rule 4 (SSoT): Internal modules are isolated to prevent bypass of state 
  validation and configuration rules.
"""

from src.step4.orchestrate_step4 import orchestrate_step4

# We restrict the public interface to the orchestrator.
# The internal io_archivist is shielded from direct external access 
# to ensure all I/O operations respect the state's lifecycle.
__all__ = ["orchestrate_step4"]