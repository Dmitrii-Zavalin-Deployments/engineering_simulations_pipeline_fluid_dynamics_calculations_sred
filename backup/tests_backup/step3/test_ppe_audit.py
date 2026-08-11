# tests/step3/test_ppe_audit.py

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.ppe_solver import solve_pressure_poisson_step


def test_solve_ppe_step_post_update_audit_failure():
    """
    Targets Lines 79-80: Triggers ArithmeticError if p_new is non-finite.
    """
    # 1. Setup a mock StencilBlock with valid geometry but "explosive" neighbors
    mock_block = MagicMock()
    mock_block.id = "block_explosive_001"
    mock_block.dx = 0.1
    mock_block.dy = 0.1
    mock_block.dz = 0.1
    mock_block.dt = 0.01
    
    # Provide valid finite values for initial checks
    mock_block.center.get_field.return_value = 0.0  # p_c and p_old
    mock_block.i_plus.get_field.return_value = 0.0
    mock_block.i_minus.get_field.return_value = 0.0
    mock_block.j_plus.get_field.return_value = 0.0
    mock_block.j_minus.get_field.return_value = 0.0
    mock_block.k_plus.get_field.return_value = 0.0
    mock_block.k_minus.get_field.return_value = 0.0

    # 2. Mock external dependencies to return finite values for the first half
    with patch("src.step3.ppe_solver.compute_local_divergence_v_star", return_value=0.0), \
         patch("src.step3.ppe_solver.get_rho_over_dt", return_value=100.0):
        
        # 3. Force sum_neighbors to be non-finite (Infinity) 
        # to ensure the final SOR calculation (p_new) results in np.inf
        mock_block.i_plus.get_field.side_effect = [
            0.0,    # First call for lap_p_n (line 35)
            np.inf  # Second call for sum_neighbors (line 57)
        ]

        # 4. Execute and catch the Rule 7 Post-Update failure
        with pytest.raises(ArithmeticError, match="Non-finite pressure generated in SOR step"):
            solve_pressure_poisson_step(
                block=mock_block, 
                divergence_threshold=1e6, 
                omega=1.5
            )

def test_solve_ppe_step_success():
    """
    Verifies the happy path and line 88 return value.
    """
    mock_block = MagicMock()
    mock_block.dx, mock_block.dy, mock_block.dz = 1.0, 1.0, 1.0
    mock_block.dt = 0.1
    
    # Initialize all fields to 0
    mock_block.center.get_field.return_value = 0.0
    for attr in ['i_plus', 'i_minus', 'j_plus', 'j_minus', 'k_plus', 'k_minus']:
        getattr(mock_block, attr).get_field.return_value = 0.0

    with patch("src.step3.ppe_solver.compute_local_divergence_v_star", return_value=1.0), \
         patch("src.step3.ppe_solver.get_rho_over_dt", return_value=1.0):
        
        # omega=1.0 simplifies the SOR formula: p_new = (Source + Neighbors)/Denom
        # stencil_denom = 2 * (1 + 1 + 1) = 6
        # rhs = 1.0 * (1.0 - 0.0) = 1.0
        # p_new = (0.0 - 1.0) / 6.0 = -0.1666...
        delta = solve_pressure_poisson_step(mock_block, 100.0, 1.0)
        
        assert np.isclose(delta, 1.0/6.0)
        mock_block.center.set_field.assert_called_with(FI.P_NEXT, pytest.approx(-1.0/6.0))