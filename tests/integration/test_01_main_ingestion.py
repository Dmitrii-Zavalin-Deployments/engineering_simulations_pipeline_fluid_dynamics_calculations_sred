"""
tests/integration/test_01_main_ingestion.py
Integration Test 1: Main CLI entry point -> Ingestion schema parsing and full fixture fidelity.
"""

import sys
from unittest.mock import patch

from src.ingestion import load_and_validate_inputs


def test_main_cli_ingestion_stage(workspace_folder, monkeypatch):
    """
    Narrative: Verifies the end-to-end integration of the Command Line Interface (CLI) 
    entry point with the strict ingestion and validation pipeline, ensuring absolute 
    structural and value parity between the conftest.py fixtures and ingested payloads.
    """
    
    # We retrieve the temporary workspace folder and explicit file handles established 
    # for the current simulation run by conftest.py.
    folder = workspace_folder["folder"]
    input_file = workspace_folder["input_file_name"]
    workspace_folder["config_path"]

    # We construct the simulated CLI argument vector mimicking user invocation:
    #     python main.py --input_output_folder <folder> --input_file_name <input> --output_file_name <output>
    cli_args = [
        "main.py",
        "--input_output_folder", folder,
        "--input_file_name", input_file,
        "--output_file_name", "simulation_results.zip",
    ]
    monkeypatch.setattr(sys, "argv", cli_args)

    # We initialize a telemetry capture ledger to intercept the parsed data payloads 
    # returned by the ingestion module during execution.
    captured_return = []

    def spy_wrapper(*args, **kwargs):
        # The ingestion pipeline processes raw files into deterministic data dictionaries:
        #     res = (input_data, config_data)
        res = load_and_validate_inputs(*args, **kwargs)
        captured_return.append(res)
        return res

    # We deploy a mock spy wrapper around load_and_validate_inputs while short-circuiting 
    # downstream solver execution via SolverState to isolate the ingestion stage.
    with patch("src.main.load_and_validate_inputs", side_effect=spy_wrapper) as spy_ingest:
        with patch("src.main.SolverState") as mock_state_cls:
            # Execution terminates cleanly after state instantiation with exit code 0:
            #     SystemExit(0)
            mock_state_cls.side_effect = SystemExit(0)
            try:
                from src.main import main
                main()
            except SystemExit as e:
                # The expected exit code for a successful early-stage interception is:
                #     e.code == 0
                assert e.code == 0

        # We verify that the ingestion spy was successfully invoked exactly once,
        # and that every single dictionary component matches the reference fixtures.
        assert spy_ingest.called
        assert len(captured_return) == 1
        
        input_data, config_data = captured_return[0]

        # 1. Physical Constraints Verification:
        #     min_velocity = -10.0, max_velocity = 10.0
        #     min_pressure = -100.0, max_pressure = 100.0
        assert input_data["physical_constraints"]["min_velocity"] == -10.0
        assert input_data["physical_constraints"]["max_velocity"] == 10.0
        assert input_data["physical_constraints"]["min_pressure"] == -100.0
        assert input_data["physical_constraints"]["max_pressure"] == 100.0

        # 2. Domain Configuration Verification:
        #     type = "INTERNAL", reference_velocity = [0.0, 0.0, 0.0]
        assert input_data["domain_configuration"]["type"] == "INTERNAL"
        assert input_data["domain_configuration"]["reference_velocity"] == [0.0, 0.0, 0.0]

        # 3. Grid Discretization Verification (4x4x4 domain):
        #     nx = 4, ny = 4, nz = 4
        assert input_data["grid"]["nx"] == 4
        assert input_data["grid"]["ny"] == 4
        assert input_data["grid"]["nz"] == 4
        assert input_data["grid"]["x_min"] == 0.0
        assert input_data["grid"]["x_max"] == 1.0

        # 4. Fluid Properties Verification:
        #     density = 1.0, viscosity = 0.01
        assert input_data["fluid_properties"]["density"] == 1.0
        assert input_data["fluid_properties"]["viscosity"] == 0.01

        # 5. Initial Conditions Verification:
        assert input_data["initial_conditions"]["velocity"] == [0.0, 0.0, 0.0]
        assert input_data["initial_conditions"]["pressure"] == 0.0

        # 6. Simulation Parameters Verification:
        #     time_step = 0.001, total_time = 0.003, output_interval = 1
        assert input_data["simulation_parameters"]["time_step"] == 0.001
        assert input_data["simulation_parameters"]["total_time"] == 0.003
        assert input_data["simulation_parameters"]["output_interval"] == 1

        # 7. Boundary Conditions Verification:
        assert len(input_data["boundary_conditions"]) == 1
        assert input_data["boundary_conditions"][0]["location"] == "wall"
        assert input_data["boundary_conditions"][0]["type"] == "no-slip"

        # 8. Mask Array Verification (length nx * ny * nz = 64):
        assert len(input_data["mask"]) == 64
        assert input_data["mask"] == [0] * 64

        # 9. External Forces Verification:
        assert input_data["external_forces"]["force_vector"] == [0.0, 0.0, 0.0]
        assert input_data["external_forces"]["gravity_vector"] == [0.0, -9.81, 0.0]

        # 10. Numerical Configuration Parameters Verification:
        #     max_poisson_iterations = 500, poisson_tolerance = 1e-6
        assert config_data["max_poisson_iterations"] == 500
        assert config_data["poisson_tolerance"] == 1e-6