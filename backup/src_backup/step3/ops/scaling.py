# src/step3/ops/scaling.py

import logging

import numpy as np

from src.common.stencil_block import StencilBlock

# Rule 7: Granular Traceability for Numerical Scaling
logger = logging.getLogger("Solver.Ops.Scaling")

def get_dt_over_rho(block: StencilBlock) -> float:
    """
    Returns the scaling factor (dt / rho) for the Predictor and Corrector Steps.
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Guards against vacuum density (rho=0).
    - Rule 4 (SSoT): Derived from immutable block properties.
    """
    # Rule 8: Contract Violation Check (Density cannot be zero or negative)
    if block.rho <= 0:
        logger.critical(
            f"PHYSICS CRASH: Block {block.id} has invalid density (rho={block.rho}). "
            "Vacuum or negative density would cause infinite acceleration."
        )
        raise ValueError(f"Invalid rho={block.rho} in block {block.id}")

    scaling = block.dt / block.rho

    # Forensic Audit
    if not np.isfinite(scaling):
        logger.error(f"MATH FAILURE: dt/rho exploded in {block.id} (dt={block.dt}, rho={block.rho})")
        raise ArithmeticError(f"Scaling factor dt/rho is non-finite in block {block.id}")

    return scaling

def get_rho_over_dt(block: StencilBlock) -> float:
    """
    Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
    
    Compliance:
    - Rule 7: Fail-Fast math audit. Guards against dt=0.
    - Zero-Debt Mandate: Explicit arithmetic; no intermediate caching.
    """
    # Rule 8: Contract Violation Check (Time step cannot be zero)
    if block.dt <= 0:
        logger.critical(
            f"TEMPORAL CRASH: Block {block.id} has invalid time-step (dt={block.dt}). "
            "Zero dt would cause infinite pressure source terms."
        )
        raise ZeroDivisionError(f"Invalid dt={block.dt} in block {block.id}")

    scaling = block.rho / block.dt

    # Forensic Audit
    if not np.isfinite(scaling):
        logger.error(f"MATH FAILURE: rho/dt exploded in {block.id} (rho={block.rho}, dt={block.dt})")
        raise ArithmeticError(f"Scaling factor rho/dt is non-finite in block {block.id}")

    return scaling