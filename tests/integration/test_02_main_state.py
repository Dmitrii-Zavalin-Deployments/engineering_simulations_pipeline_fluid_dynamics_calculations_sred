"""
tests/integration/test_02_main_state.py
Integration Test 2: Main CLI -> Ingestion -> Sovereign SolverState initialization and comprehensive state fidelity verification.
"""

import sys
from unittest.mock import patch

from src.state import SolverState


def test_main_cli_state_stage(workspace_folder, monkeypatch):
    """
    Narrative: Verifies the sovereign integration pipeline from command-line parsing
    through strict ingestion to sovereign SolverState initialization, ensuring absolute
    structural and value parity of every fixture attribute inside the instantiated SolverState object.
    """
    
    # We retrieve the temporary workspace folder path and input file handle 
    # for the current simulation run established by conftest.py.
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]

    # We configure the simulated command-line arguments:
    #     python main.py --input_output_folder <folder> --input_file_name <input> --output_file_name <output>
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", "simulation_results.zip",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # We establish a telemetry capture ledger for the instantiated sovereign SolverState object.
    captured_state = []

    def state_capture_wrapper(input_data, config_data):
        # The sovereign SolverState orchestrates physical simulation matrices from ingested data:
        #     instance = SolverState(input_data, config_data)
        instance = SolverState(input_data, config_data)
        captured_state.append(instance)
        # We short-circuit execution right after state creation:
        #     SystemExit(0)
        raise SystemExit(0)

    # We patch SolverState to intercept state initialization during main execution.
    with patch("src.main.SolverState", side_effect=state_capture_wrapper):
        try:
            from src.main import main
            main()
        except SystemExit as e:
            # The expected exit code for the intercepted sovereign state stage is:
            #     e.code == 0
            assert e.code == 0

    # We verify that exactly one state instance was captured, and inspect its structural properties.
    assert len(captured_state) == 1
    state = captured_state[0]
    
    # For the multi-dimensional simulation grid (nx=4, ny=4, nz=4) and field tensor components, 
    # the expected tensor shape is:
    #     fields.shape = (4, 4, 4, 4)
    assert state.fields.shape == (4, 4, 4, 4)

    # For simulation time parameters defined in conftest.py:
    #     total_time = 0.003
    #     time_step = 0.001
    # The expected iteration count computed via:
    #     total_iterations = total_time / time_step = 3
    assert state.total_iterations == 3

    # We perform comprehensive validation confirming that every single input and config element 
    # from the reference conftest fixtures is correctly stored within the SolverState object:
    
    # 1. Physical Constraints Verification:
    #     min_velocity = -10.0, max_velocity = 10.0
    #     min_pressure = -100.0, max_pressure = 100.0
    assert state.input_data["physical_constraints"]["min_velocity"] == -10.0
    assert state.input_data["physical_constraints"]["max_velocity"] == 10.0
    assert state.input_data["physical_constraints"]["min_pressure"] == -100.0
    assert state.input_data["physical_constraints"]["max_pressure"] == 100.0

    # 2. Domain Configuration Verification:
    #     type = "INTERNAL", reference_velocity = [0.0, 0.0, 0.0]
    assert state.input_data["domain_configuration"]["type"] == "INTERNAL"
    assert state.input_data["domain_configuration"]["reference_velocity"] == [0.0, 0.0, 0.0]

    # 3. Grid Discretization Verification (4x4x4 domain):
    #     nx = 4, ny = 4, nz = 4
    assert state.input_data["grid"]["nx"] == 4
    assert state.input_data["grid"]["ny"] == 4
    assert state.input_data["grid"]["nz"] == 4
    assert state.input_data["grid"]["x_min"] == 0.0
    assert state.input_data["grid"]["x_max"] == 1.0

    # 4. Fluid Properties Verification:
    #     density = 1.0, viscosity = 0.01
    assert state.input_data["fluid_properties"]["density"] == 1.0
    assert state.input_data["fluid_properties"]["viscosity"] == 0.01

    # 5. Initial Conditions Verification:
    assert state.input_data["initial_conditions"]["velocity"] == [0.0, 0.0, 0.0]
    assert state.input_data["initial_conditions"]["pressure"] == 0.0

    # 6. Simulation Parameters Verification:
    #     time_step = 0.001, total_time = 0.003, output_interval = 1
    assert state.input_data["simulation_parameters"]["time_step"] == 0.001
    assert state.input_data["simulation_parameters"]["total_time"] == 0.003
    assert state.input_data["simulation_parameters"]["output_interval"] == 1

    # 7. Boundary Conditions Verification:
    assert len(state.input_data["boundary_conditions"]) == 1
    assert state.input_data["boundary_conditions"][0]["location"] == "wall"
    assert state.input_data["boundary_conditions"][0]["type"] == "no-slip"

    # 8. Mask Array Verification (length nx * ny * nz = 64):
    assert len(state.input_data["mask"]) == 64
    assert state.input_data["mask"] == [0] * 64

    # 9. External Forces Verification:
    assert state.input_data["external_forces"]["force_vector"] == [0.0, 0.0, 0.0]
    assert state.input_data["external_forces"]["gravity_vector"] == [0.0, -9.81, 0.0]

    # 10. Numerical Configuration Parameters Verification:
    #     max_poisson_iterations = 500, poisson_tolerance = 1e-6
    assert state.config["max_poisson_iterations"] == 500
    assert state.config["poisson_tolerance"] == 1e-6