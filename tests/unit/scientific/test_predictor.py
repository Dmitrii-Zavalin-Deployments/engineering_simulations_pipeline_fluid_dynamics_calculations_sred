# tests/scientific/test_predictor.py

import logging

import numpy as np
import pytest

from src.common.field_schema import FI
from src.step3.predictor import compute_local_predictor_step
from tests.helpers.solver_step2_output_dummy import make_step2_output_dummy


def setup_predictor_block(dt=1.0, rho=1.0, mu=1.0, dx=1.0):
    """Wires a StencilBlock with unit physics for analytical transparency."""
    state = make_step2_output_dummy(nx=4, ny=4, nz=4)
    block = state.stencil_matrix[10]  # Central core block

    # Standardize physics to unit values for clean math verification
    params = {
        '_dx': float(dx),
        '_dy': float(dx),
        '_dz': float(dx),
        '_dt': float(dt),
        '_rho': float(rho),
        '_mu': float(mu),
        '_f_vals': (0.0, 0.0, 0.0),
    }
    for attr, val in params.items():
        object.__setattr__(block, attr, val)

    # Clean the buffer to ensure no residual data contamination
    block.center.fields_buffer.fill(0.0)
    return block


# --- PHYSICS INTEGRATION TESTS ---


def test_predictor_diffusion_only():
    """Verifies: v* = v_n + (dt/rho) * (mu * lap(v_n))

    Setup: lap=2.0, all other terms 0.
    Expectation: 0 + (1/1) * (1.0 * 2.0) = 2.0
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0)

    # Curvature in X: (ip: 1.0, c: 0.0, im: 1.0) -> lap = 2.0
    block.i_plus.set_field(FI.VX, 1.0)
    block.i_minus.set_field(FI.VX, 1.0)

    compute_local_predictor_step(block)

    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(
        2.0
    ), f"Diffusion failed: expected 2.0, got {obtained}"


def test_predictor_full_3d_complex_integration():
    """Full Equation: v* = v_n + (dt/rho) * [ (mu * lap) - (rho * adv) + F - grad_p ]

    Setup:
    - v_n = 1.0
    - lap = 0.0 (Linear slope)
    - adv = 1.0 (u * du/dx = 1 * 1)
    - grad_p = 1.0
    - F = 0.0
    Calculation: 1.0 + (1/1) * [ (1*0) - (1*1) + 0 - 1 ] = -1.0
    """
    block = setup_predictor_block(mu=1.0, dt=1.0, rho=1.0, dx=1.0)

    # 1. Velocity slope: (im:0, c:1, ip:2) -> grad_v=1, lap_v=0
    block.center.set_field(FI.VX, 1.0)
    block.i_plus.set_field(FI.VX, 2.0)
    block.i_minus.set_field(FI.VX, 0.0)

    # 2. Match Y/Z neighbors to center to keep Laplacian 1D
    for n in [block.j_plus, block.j_minus, block.k_plus, block.k_minus]:
        n.set_field(FI.VX, 1.0)

    # 3. Pressure gradient: (ip:2, im:0) -> grad_p = 1.0
    block.i_plus.set_field(FI.P, 2.0)
    block.i_minus.set_field(FI.P, 0.0)

    compute_local_predictor_step(block)

    obtained = block.center.get_field(FI.VX_STAR)
    assert obtained == pytest.approx(
        -1.0
    ), f"Complex integration failed: expected -1.0, got {obtained}"


# --- FORENSIC LOGGING & AUDIT TESTS ---


def test_predictor_audit_logging(caplog):
    """Verifies AUDIT debug logs are generated for operator outputs."""
    block = setup_predictor_block()

    with caplog.at_level(logging.DEBUG):
        compute_local_predictor_step(block)

    assert "DEBUG [Predictor]: Type=Sovereign" in caplog.text
    assert "PREDICT [Success]" in caplog.text
    assert "Type=" in caplog.text


def test_predictor_component_info_logging(caplog):
    """Verifies VX_STAR value is logged at DEBUG level during computation."""
    print("\n--- [START] test_predictor_component_info_logging ---")

    block = setup_predictor_block()

    # Silence memory noise to prevent log pollution
    logging.getLogger("src.common.solver_state").setLevel(logging.ERROR)

    with caplog.at_level(logging.DEBUG, logger="Solver.Predictor"):
        compute_local_predictor_step(block)

    log_text = caplog.text
    print(f"DEBUG [Test]: Full log output:\n{log_text}")

    # Assertions
    assert "DEBUG [Predictor]: Type=Sovereign" in log_text
    assert "VX_STAR:" in log_text
    assert "PREDICT [Success]:" in log_text

    print("--- [END] test_predictor_component_info_logging ---\n")


def test_predictor_contamination_recovery(caplog):
    """Verifies Rule 7 & Rule 3: DNA Audit.

    Note: In the new architecture, the Predictor trusts the Laplacian's
    internal float-enforcement. This test now verifies that the Predictor
    correctly handles/logs failures propagated from corrupted operators.
    """
    print("\n--- [START] test_predictor_contamination_recovery ---")

    block = setup_predictor_block()

    # 1. Access the predictor module for monkeypatching
    from src.step3 import predictor

    original_lap = predictor.compute_local_laplacian_v_n

    # 2. Inject a Non-Finite value to trigger the Predictor's internal Fail-Fast Audit
    # Since we removed the "is it an array" check from the Predictor, we test
    # its ability to catch 'Numerical Contamination' (NaN).
    contaminant = (np.nan, 0.0, 0.0)
    predictor.compute_local_laplacian_v_n = lambda b: contaminant
    print(f"DEBUG [Test]: Monkeypatch injected | Value: {contaminant}")

    try:
        # 3. Predictor must log CRITICAL and raise ArithmeticError on NaN
        with caplog.at_level(logging.CRITICAL, logger="Solver.Predictor"):
            print(
                "DEBUG [Test]: Calling compute_local_predictor_step (expecting"
                " failure)..."
            )
            with pytest.raises(ArithmeticError):
                compute_local_predictor_step(block)

        # 4. Forensic Verification
        log_content = caplog.text
        has_failure_msg = "PREDICTOR FAILURE" in log_content
        print(
            "DEBUG [Test]: Search for 'PREDICTOR FAILURE' | Found:"
            f" {has_failure_msg}"
        )

        assert (
            has_failure_msg
        ), "Predictor failed to log CRITICAL on numerical contamination."

    finally:
        predictor.compute_local_laplacian_v_n = original_lap
        print(
            "DEBUG [Test]: Restored original_lap. Test isolation preserved."
        )
        print("--- [END] test_predictor_contamination_recovery ---\n")


def test_predictor_math_failure_traceback(caplog):
    """Verifies Rule 7: Sovereign Entry Logging (Forensic Trace)."""
    print("\n--- [START] test_predictor_math_failure_traceback ---")

    block = setup_predictor_block()

    # 1. Sabotage the block - remove 'mu' (viscosity)
    # This triggers an AttributeError when the Predictor tries to calculate v_star.
    if hasattr(block, "mu"):
        try:
            del block.mu
        except AttributeError:
            delattr(block, "_mu") if hasattr(block, "_mu") else None

    print(
        "DEBUG [Test]: Sabotaged block | mu check: "
        + str(hasattr(block, "mu"))
    )

    # 2. Execution Context
    print("DEBUG [Test]: Entering caplog context (Level: DEBUG)")
    with caplog.at_level(
        logging.DEBUG, logger="Solver.Predictor"
    ), pytest.raises(AttributeError):
        compute_local_predictor_step(block)

    # 3. Forensic Log Audit
    log_text = caplog.text
    search_string = "DEBUG [Predictor]: Type=Sovereign"
    found_marker = search_string in log_text

    print(
        f"DEBUG [Test]: Searching for '{search_string}' | Found: {found_marker}"
    )

    if not found_marker:
        print("DEBUG [Test]: DUMPING FULL LOG FOR ANALYSIS:")
        print(log_text if log_text else "[EMPTY LOG]")

    assert (
        found_marker
    ), "Forensic log missing! Sovereign entry must be logged before attribute access."
    print("--- [END] test_predictor_math_failure_traceback ---\n")
