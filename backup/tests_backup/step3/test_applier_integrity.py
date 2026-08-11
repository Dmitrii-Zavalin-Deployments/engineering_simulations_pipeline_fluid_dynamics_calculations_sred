# tests/step3/test_applier_integrity.py

import logging
import math

import numpy as np
import pytest

from src.common.cell import Cell
from src.common.field_schema import FI
from src.common.stencil_block import StencilBlock
from src.step3.boundaries.applier import apply_boundary_values

# Rule 7: Granular Traceability
logger = logging.getLogger(__name__)


def create_real_applier_block(block_id="TestBlock_01"):
    """
    Rule 9: Creates a real StencilBlock with a dedicated NumPy buffer.
    Fixes the 'id' setter error by using object.__setattr__ for slotted properties.
    """
    # Pre-allocate buffer for all fields (1 row, matching FI.num_fields)
    buffer = np.zeros((1, FI.num_fields()))

    center = Cell(
        index=0,
        fields_buffer=buffer,
        nx_buf=3,
        ny_buf=3,
        is_ghost=True,  # Applier usually works on Ghost cells
    )

    # Minimal block setup to satisfy Rule 5
    # Neighbors are None because Applier only mutates 'center'
    nb = [None] * 6
    block = StencilBlock(
        center=center,
        i_minus=nb[0],
        i_plus=nb[1],
        j_minus=nb[2],
        j_plus=nb[3],
        k_minus=nb[4],
        k_plus=nb[5],
        dx=0.1,
        dy=0.1,
        dz=0.1,
        dt=0.01,
        rho=1.0,
        mu=0.01,
        f_vals=(0, 0, 0),
    )

    # FIX: Since 'id' is a read-only property using __slots__,
    # we must mutate the underlying private attribute '_id'.
    object.__setattr__(block, "_id", block_id)

    return block


def test_applier_enforcement_real_memory(caplog):
    """
    VERIFICATION: Ensure the Applier writes 1e10 to the real VX_STAR buffer
    and generates the Rule 7 forensic audit trail.
    """
    block = create_real_applier_block("ExplosionBlock_99")

    # u maps to VX_STAR per BC_FIELD_MAP in applier.py
    rule = {
        "location": "x_min",
        "type": "inflow",
        "values": {"u": 1.0e10},
    }

    with caplog.at_level(logging.DEBUG):
        apply_boundary_values(block, rule)

    # A. Physical Memory Check (Rule 9)
    # Extract scalar from numpy buffer for comparison
    actual_val = block.center.fields_buffer[0, FI.VX_STAR]
    assert math.isclose(actual_val, 1e10, rel_tol=1e-12)

    # B. Forensic Audit Trail (Rule 7)
    assert "APPLY [Success]: Block ExplosionBlock_99" in caplog.text
    assert f"Mapping u to Field {int(FI.VX_STAR)}" in caplog.text
    assert (
        "APPLY [Success]: Block ExplosionBlock_99 | Location: x_min | Field"
        " VX_STAR set to 1.0000e+10"
        in caplog.text
    )


def test_applier_unsupported_key_contract_violation(caplog):
    """Rule 8: Invalid keys must trigger a Contract Violation and KeyError."""
    block = create_real_applier_block("ErrorBlock_01")

    invalid_rule = {
        "location": "x_max",
        "type": "outlet",
        "values": {"dark_matter_density": 99.9},
    }

    with caplog.at_level(logging.ERROR), pytest.raises(
        KeyError, match="Unsupported boundary key"
    ):
        apply_boundary_values(block, invalid_rule)

    assert (
        "CONTRACT VIOLATION: Unsupported key 'dark_matter_density'"
        in caplog.text
    )


def test_applier_missing_metadata_strategic_failure(caplog):
    """Verify Rule 5: Fail fast if rule metadata is incomplete."""
    block = create_real_applier_block("BrokenBlock")

    broken_rule = {
        "location": "y_min",
        "values": {"p": 101325.0},  # Missing 'type'
    }

    with caplog.at_level(logging.ERROR), pytest.raises(
        ValueError, match="Boundary rule missing critical fields"
    ):
        apply_boundary_values(block, broken_rule)

    assert "STRATEGIC FAILURE" in caplog.text


def test_applier_p_mapping_precision(caplog):
    """Verify that 'p' maps correctly to P_NEXT (Field 7) with machine precision."""
    block = create_real_applier_block("PressureBlock")
    rule = {
        "location": "z_max",
        "type": "dirichlet",
        "values": {"p": 50.5},
    }

    apply_boundary_values(block, rule)

    # Check that P_NEXT was updated (Field 7), NOT the current P (Field 6)
    assert block.center.get_field(FI.P_NEXT) == 50.5
    assert block.center.get_field(FI.P) == 0.0
