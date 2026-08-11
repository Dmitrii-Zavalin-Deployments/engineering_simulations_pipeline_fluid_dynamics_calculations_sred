# tests/quality_gates/sensitivity_gate/test_stencil_fidelity.py

import numpy as np
import pytest

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.stencil_assembler import assemble_stencil_matrix
from tests.helpers.solver_input_schema_dummy import create_validated_input


def test_gate_2a_registry_traceability():
    """
    Gate 2.A: Stencil Registry Traceability Audit
    Identity: Constants must propagate from JSON to Cell objects.
    Compliance: Rule 4 (SSoT Hierarchy)
    """
    # 1. Setup: create_validated_input returns the raw SolverInput
    nx, ny, nz = 2, 2, 2
    solver_input = create_validated_input() 
    
    # Pathing Fix: Align with Rule 4 Hierarchy
    solver_input.grid.nx = nx
    solver_input.grid.ny = ny
    solver_input.grid.nz = nz

    # Traceable constants must live in fluid_properties sub-container
    traceable_rho = 13.52
    traceable_mu = 0.00173
    solver_input.fluid_properties.density = traceable_rho
    solver_input.fluid_properties.viscosity = traceable_mu
    
    # Compliance: Wrap in SimulationContext to satisfy the Firewall Mandate
    context = SimulationContext(input_data=solver_input, config=None)
    
    # 2. Action: Hydrate through the actual orchestrator
    state = orchestrate_step1(context)
    stencils = assemble_stencil_matrix(state)
    
    # 3. Verification: Registry Traceability (Success Metric)
    # Constants must reach the Logic Layer (Cell objects) from the JSON intake.
    sample_block = stencils[0]
    assert np.isclose(sample_block.rho, traceable_rho), (
        f"Traceability Failure: rho {sample_block.rho} != {traceable_rho}"
    )
    assert np.isclose(sample_block.mu, traceable_mu), (
        f"Traceability Failure: mu {sample_block.mu} != {traceable_mu}"
    )

def test_gate_2a_foundation_mismatch_catch(monkeypatch):
    """
    Verification: Catch RuntimeError when buffer width deviates from FI Schema.
    Ensures Topology Protection before Step 2 assembly.
    """
    solver_input = create_validated_input()
    solver_input.grid.nx = 2
    solver_input.grid.ny = 2
    solver_input.grid.nz = 2
    
    context = SimulationContext(input_data=solver_input, config=None)
    state = orchestrate_step1(context)
    
    # Corrupt the Foundation: Width mismatch (Schema Lock)
    required_width = FI.num_fields()
    bad_width = required_width - 1
    # Rule 9: (nx+2)^3 = 4^3 = 64 cells
    corrupted_data = np.zeros((64, bad_width))
    
    monkeypatch.setattr(state.fields, "data", corrupted_data)
    
    # Verification: The Registry Assembler must reject the foundation mismatch
    with pytest.raises(RuntimeError, match="Foundation Mismatch"):
        assemble_stencil_matrix(state)

def test_gate_2a_registry_is_read_only():
    """Rule 4: Geometric Context must be read-only during assembly."""
    solver_input = create_validated_input()
    context = SimulationContext(input_data=solver_input, config=None)
    
    state = orchestrate_step1(context)
    original_nx = state.grid.nx
    
    # Action: Assembly should never mutate the input geometric state
    _ = assemble_stencil_matrix(state)
    
    assert state.grid.nx == original_nx, (
        "Rule 4 Breach: Stencil assembly mutated the Geometric Context."
    )