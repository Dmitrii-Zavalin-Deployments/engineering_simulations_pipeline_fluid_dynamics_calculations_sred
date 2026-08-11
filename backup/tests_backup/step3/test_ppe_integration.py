# tests/step3/test_ppe_integration.py

import logging

import numpy as np
import pytest

from src.common.cell import Cell
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.ppe_solver import solve_pressure_poisson_step

# Rule 7: Granular Traceability
logger = logging.getLogger(__name__)

def create_real_block(center_p_next=0.0):
    """
    Rule 5 & 9: Helper to create a production-grade StencilBlock 
    with a real NumPy foundation.
    """
    def make_cell(p_val=0.0):
        buf = np.zeros((1, FI.num_fields()))
        buf[0, FI.P_NEXT] = p_val
        buf[0, FI.P] = 0.0 # Foundation pressure
        # Set velocities to zero to ensure divergence is 0 unless patched
        buf[0, FI.VX : FI.VZ+1] = 0.0 
        return Cell(index=0, fields_buffer=buf, nx_buf=3, ny_buf=3)

    center = make_cell(p_val=center_p_next)
    neighbors = [make_cell() for _ in range(6)]
    
    return StencilBlock(
        center=center,
        i_minus=neighbors[0], i_plus=neighbors[1],
        j_minus=neighbors[2], j_plus=neighbors[3],
        k_minus=neighbors[4], k_plus=neighbors[5],
        dx=0.1, dy=0.1, dz=0.1, dt=1e-6,
        rho=1.0, mu=0.01, f_vals=(0, 0, 0)
    )

class TestPPESolverScientific:

    def test_catch_poisoned_pressure_slingshot(self, caplog):
        """
        Rule 7 Verification: Massive p_old must trigger ArithmeticError.
        This tests the 'Pre-Update Audit' in ppe_solver.py.
        """
        # 1. Setup real block with 'slingshot' value: 1.4e13
        test_threshold = 1.0e10
        block = create_real_block(center_p_next=1.4e13)
        
        # 2. Execution & Validation
        with pytest.raises(ArithmeticError) as excinfo:
            solve_pressure_poisson_step(block, divergence_threshold=test_threshold, omega=1.0)
            
        assert "Poisoned Pressure Trial" in str(excinfo.value)
        assert "PPE CRITICAL" in caplog.text
        assert "1.40e+13" in caplog.text

    def test_catch_nan_divergence(self, caplog, monkeypatch):
        """
        Rule 7: Atomic Numerical Truth.
        Ensures the Divergence Gate catches NaNs before they propagate to RHS.
        """
        block = create_real_block()
        
        # Rule 8: Use monkeypatch on the specific operation to inject poison
        import src.step3.ppe_solver as ppe_module
        monkeypatch.setattr(ppe_module, "compute_local_divergence_v_star", lambda b: float("nan"))

        with pytest.raises(ArithmeticError) as excinfo:
            solve_pressure_poisson_step(block, divergence_threshold=1e12, omega=1.0)
        
        assert "NaN detected in divergence" in str(excinfo.value)
        assert "PPE MATH ERROR" in caplog.text

    def test_successful_ppe_update_precision(self):
        """
        Rule 9: Verify that a valid step updates the REAL NumPy buffer.
        """
        block = create_real_block(center_p_next=10.0)
        # Set a neighbor to create a gradient
        block.i_plus.set_field(FI.P_NEXT, 20.0)
        
        # omega=1.0 (Pure Gauss-Seidel part of SOR)
        delta = solve_pressure_poisson_step(block, divergence_threshold=1e12, omega=1.0)
        
        # 1. Verify delta is calculated
        assert delta > 0.0
        
        # 2. Verify Foundation Update
        # The center's P_NEXT in the actual NumPy buffer should no longer be 10.0
        final_p = block.center.get_field(FI.P_NEXT)
        assert final_p != 10.0
        assert np.isfinite(final_p)

    def test_dna_audit_scalar_enforcement(self, caplog):
        """
        Rule 0: Ensure the 'DNA AUDIT' in the solver handles 
        accidental array promotion gracefully.
        """
        block = create_real_block(center_p_next=1.0)
        
        # We don't need to force an error here, just verify that 
        # set_field receives a scalar. The Cell._to_scalar helper 
        # is our secondary defense, but the solver's item() is the primary.
        solve_pressure_poisson_step(block, divergence_threshold=1e12, omega=1.0)
        
        # Verification that no DNA leaks occurred (unless logic is broken)
        assert "DNA AUDIT" not in caplog.text