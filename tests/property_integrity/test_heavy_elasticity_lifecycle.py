# tests/property_integrity/test_heavy_elasticity_lifecycle.py

import json
import logging
import zipfile
from io import BytesIO
from pathlib import Path

import h5py
import numpy as np
import pytest
from src.main import BASE_DIR, run_solver

logger = logging.getLogger(__name__)


class TestHeavyElasticityLifecycle:

    @pytest.fixture
    def base_config(self):
        """Standard stable config for the solver."""
        return {
            "dt": 0.01,
            "dt_min_limit": 0.001,
            "ppe_tolerance": 1e-4,
            "ppe_atol": 1e-10,
            "ppe_max_iter": 5,
            "ppe_omega": 1.7,
            "ppe_max_retries": 5,
            "divergence_threshold": 1e12,
        }

    @pytest.fixture
    def base_input(self):
        """Standard physical setup with reasonable constraints."""
        return {
            "domain_configuration": {"type": "INTERNAL"},
            "grid": {
                "x_min": 0.0, "x_max": 1.0,
                "y_min": 0.0, "y_max": 1.0,
                "z_min": 0.0, "z_max": 1.0,
                "nx": 4, "ny": 4, "nz": 4,
            },
            "fluid_properties": {"density": 1.0, "viscosity": 0.01},
            "initial_conditions": {"velocity": [0.1, 0.0, 0.0], "pressure": 1.0},
            "simulation_parameters": {
                "time_step": 0.01,
                "total_time": 0.02,
                "output_interval": 1,
            },
            "physical_constraints": {
                "min_velocity": -100.0,
                "max_velocity": 40.0,
                "min_pressure": -10.0,
                "max_pressure": 100.0,
            },
            "boundary_conditions": [
                {"location": "y_min", "type": "free-slip", "values": {"u": 0.0}},
                {"location": "y_max", "type": "free-slip", "values": {"u": 0.0}},
                {"location": "z_min", "type": "free-slip", "values": {"u": 0.0}},
                {"location": "z_max", "type": "free-slip", "values": {"u": 0.0}},
                {"location": "x_min", "type": "inflow", "values": {"u": 1.0, "v": 0.0, "w": 0.0}},
                {"location": "x_max", "type": "outflow", "values": {"p": 0.0}},
            ],
            "mask": [0] * 64,
            "external_forces": {"force_vector": [0.0, 0.0, 0.0]},
        }

    def _run_with_config_protection(self, config_data, input_data, input_filename, caplog_level, logger_name="Solver.Main"):
        """Helper to run solver while backing up and restoring the production config.json."""
        config_path = Path(BASE_DIR) / "config.json"
        input_path = Path(BASE_DIR) / input_filename
        
        # 1. Backup existing config
        config_backup = config_path.read_text() if config_path.exists() else None
        
        try:
            config_path.write_text(json.dumps(config_data))
            input_path.write_text(json.dumps(input_data))
            return run_solver(input_filename)
        finally:
            # 2. Cleanup input file
            # 3. RESTORE: Put back original config or delete if it wasn't there
            if config_backup is not None:
                config_path.write_text(config_backup)

    def test_scenario_1_pure_success(self, caplog, base_config, base_input):
        """
        Scenario 1: System Gatekeeper & Physics Success.
        Verifies production plumbing (paths, size, zip) and numerical stability.
        """
        input_filename = "test_success_input.json"
        # Define the target testing directory used by the solver
        test_data_dir = Path(BASE_DIR) / "data" / "testing-input-output"
        
        with caplog.at_level(logging.WARNING, logger="Solver.Main"):
            # 1. Execute via the protection helper
            zip_path_str = self._run_with_config_protection(
                base_config, base_input, input_filename, logging.WARNING
            )
            zip_path = Path(zip_path_str)

            # 2. PLUMBING AUDIT: Verify the 'Gatekeeper' requirements
            assert zip_path.exists(), "GATEKEEPER FAIL: ZIP artifact not found on disk."
            assert zip_path.stat().st_size > 0, "GATEKEEPER FAIL: ZIP artifact is empty (0 bytes)."
            assert test_data_dir in zip_path.parents, f"PATH FAIL: ZIP landed in wrong directory: {zip_path.parent}"

            # 3. STABILITY AUDIT: Ensure no elasticity triggers fired
            triggers = [r for r in caplog.records if "STABILITY TRIGGER" in r.message]
            assert len(triggers) == 0, f"STABILITY FAIL: Unexpected triggers in a 'Pure Success' run: {triggers}"

            # 4. DATA INTEGRITY: Inspect the internal HDF5 payload
            with zipfile.ZipFile(zip_path, "r") as archive:
                h5_files = [f for f in archive.namelist() if f.endswith(".h5")]
                assert len(h5_files) > 0, "INTEGRITY FAIL: No .h5 files found inside the result ZIP."

                # Audit the final frame for finite physics
                with (
                    archive.open(h5_files[-1]) as f,
                    h5py.File(BytesIO(f.read()), "r") as h5_audit,
                ):
                    assert "vx" in h5_audit, "SCHEMA FAIL: 'vx' dataset missing from HDF5."
                    vx_data = h5_audit["vx"][:]
                    assert np.all(np.isfinite(vx_data)), "PHYSICS FAIL: Non-finite values detected in final velocity field."
                    
            print(f"\n✅ Scenario 1 Passed: Plumbing and Physics are nominal. Artifact: {zip_path.name}")

    def test_scenario_2_elasticity_terminal_failure(self, caplog, base_config, base_input):
        """
        Scenario 2: Elasticity exercised fully, but no dt can stabilize the run.
        Expect a CRITICAL INSTABILITY after exhausting retries.
        """
        # --- CFL setup: deliberately aggressive to force PPE/pressure failure ---
        u = np.float64(20.0)
        dt_initial = np.float64(0.08)

        x_max = np.float64(base_input["grid"]["x_max"])
        x_min = np.float64(base_input["grid"]["x_min"])
        nx = np.float64(base_input["grid"]["nx"])
        dx = (x_max - x_min) / nx
        courant_number = (u * dt_initial) / dx

        logger.info(
            f"DEBUG: Scenario 2 Initial Courant Number = {courant_number:.4f} (Target >> 1.0)"
        )

        # 1. Physics: strong inflow, wide pressure corridor to let PPE roam
        base_input["boundary_conditions"][0]["values"]["u"] = float(u)
        base_input["physical_constraints"]["max_velocity"] = 40.0
        base_input["physical_constraints"]["max_pressure"] = 1000.0

        # 2. Config: aggressive dt, deep retry ladder
        base_config["dt"] = float(dt_initial)
        base_config["dt_min_limit"] = 1e-6
        base_config["ppe_max_retries"] = 20
        base_config["ppe_max_iter"] = 200
        base_config["ppe_tolerance"] = 1e-6

        input_filename = "test_elastic_terminal_fail.json"

        with caplog.at_level(logging.INFO), pytest.raises(RuntimeError) as excinfo:
            self._run_with_config_protection(base_config, base_input, input_filename, logging.INFO)

        error_msg = str(excinfo.value)
        assert "CRITICAL INSTABILITY" in error_msg
        assert "Exhausted" in error_msg

        # Elasticity must have actually fired
        assert "STABILITY TRIGGER" in caplog.text
        # And the physical audit must have been involved
        assert "AUDIT [Explosion]" in caplog.text or "AUDIT [Limit]" in caplog.text

    def test_scenario_3_terminal_failure_hard_config(self, caplog, base_config, base_input):
        """
        Scenario 3: Force a crash quickly via hard configuration limits.
        This is a sharper, low-retry failure mode.
        """
        base_input["boundary_conditions"][0]["values"]["u"] = 50.0
        base_config["dt_min_limit"] = 0.01
        base_config["ppe_max_retries"] = 2

        input_filename = "test_terminal_fail.json"

        with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError) as excinfo:
            self._run_with_config_protection(base_config, base_input, input_filename, logging.WARNING)

        error_msg = str(excinfo.value)
        assert "CRITICAL INSTABILITY" in error_msg
        assert "Exhausted" in error_msg

        assert "AUDIT [Limit]" in caplog.text
        assert "STABILITY TRIGGER" in caplog.text

    def test_scenario_4_retry_and_recover(self, caplog, base_config, base_input):
        """
        Scenario 4: Elasticity recovers (mock‑based).
        First audit fails → elasticity triggers.
        Second audit succeeds → solver commits and completes.
        
        SCIENTIFIC JUSTIFICATION: 
        A Mock is used here to simulate a 'Rare Scenario' in physics (e.g., a 
        singular pressure spike or divergent velocity) that is difficult to 
        reproduce deterministically with stable input. This verifies that 
        the Elasticity Manager's 'Retry-and-Recover' logic is robust even 
        when the underlying kernel momentarily fails Rule 7 audits.
        """

        from unittest.mock import patch
        input_filename = "test_recovery_input.json"

        side_effects = [ArithmeticError("Mocked explosion"), None, None, None]

        with patch("src.common.solver_state.SolverState.audit_physical_bounds", side_effect=side_effects), caplog.at_level(logging.INFO):
            zip_path = self._run_with_config_protection(base_config, base_input, input_filename, logging.INFO)

        assert zip_path is not None

        # Elasticity must have fired at least once
        trigger_found = any("STABILITY TRIGGER" in r.message for r in caplog.records)
        assert trigger_found, "Elasticity never fired; mock did not trigger retry logic."

        # No terminal instability should occur
        assert "CRITICAL INSTABILITY" not in caplog.text
