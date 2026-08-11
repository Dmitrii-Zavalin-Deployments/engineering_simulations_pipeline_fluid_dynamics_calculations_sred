# src/common/elasticity.py

import logging

from src.common.field_schema import FI
from src.common.solver_state import SolverState


class ElasticManager:
    """
    Rule 4 & 5: SSoT Stability Controller (Elasticity Engine).
    Manages the 'safety ladder' of time-steps and enforces physical audits
    before trial data is committed to the foundation.
    """
    __slots__ = ['_dt', '_dt_range', '_iteration', '_runs', '_state', 'config', 'dt_floor', 'logger']

    def __init__(self, config, state: SolverState):
        self.config = config
        self._state = state
        self.logger = logging.getLogger("Solver.Main")
        
        # Rule 5: Pulling initial conditions from the SSoT objects
        # Using the state's internal dt as the starting anchor
        self._dt = state.simulation_parameters.time_step 
        self.dt_floor = self.config.dt_min_limit
        self._iteration = 0
        self._runs = self.config.ppe_max_retries
        
        # Linear safety ladder: from current dt down to dt_floor
        self._dt_range = [
            self._dt + i * (self.dt_floor - self._dt) / self._runs 
            for i in range(self._runs + 1)
        ]

    @property
    def dt(self) -> float: 
        """Returns the currently active time-step."""
        return self._dt

    def validate_and_commit(self) -> None:
        """
        Rule 9: Unified Data Commitment.
        Rule 7: Fail-Fast Physical Audit.
        
        This method acts as a logical gate. If the audit fails, the 
        ArithmeticError halts the process before any data is overwritten.
        """
        # 1. THE AUDIT: Vectorized bounds check via SolverState
        self._state.audit_physical_bounds()

        # 2. THE COMMIT: Bulk transfer Trial (_STAR) -> Foundation
        # Optimized via Foundation-level indexing to maintain O(N) scaling.
        data = self._state.fields.data 
        
        data[:, FI.VX] = data[:, FI.VX_STAR]
        data[:, FI.VY] = data[:, FI.VY_STAR]
        data[:, FI.VZ] = data[:, FI.VZ_STAR]
        data[:, FI.P] = data[:, FI.P_NEXT]
        self._state.iteration += 1
        self._state.time += self._dt

    def stabilization(self, is_needed: bool) -> None:
        """
        Orchestrates time-step recovery and data commitment.
        
        If is_needed is False, we attempt to commit the current step.
        If is_needed is True, we descend the 'safety ladder' to a smaller dt.
        """
        if not is_needed:
            # Attempt to commit. Audit failure here propagates an ArithmeticError.
            self.validate_and_commit()

            # Success: Reset the tension and return to full speed
            self._iteration = 0
            self._dt = self._dt_range[self._iteration]
            return

        # Recovery Logic: Check if we've hit the floor
        if self._iteration >= self._runs:
            raise RuntimeError(
                f"CRITICAL INSTABILITY: Exhausted {self._runs} retries. "
                f"Floor reached: {self.dt_floor:.2e}. Check physics/mesh parameters."
            )
        
        # Descend the ladder to the next (smaller) time step
        self._iteration += 1
        self._dt = self._dt_range[self._iteration]
        
        # Rule 8: Singular logging call for audit trail visibility
        self.logger.warning(
            f"STABILITY TRIGGER: Reducing dt to {self._dt:.2e} "
            f"({self._iteration}/{self._runs})"
        )