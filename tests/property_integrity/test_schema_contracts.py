# tests/property_integrity/test_schema_contracts.py

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from tests.helpers.solver_input_schema_dummy import create_validated_input
from tests.helpers.solver_output_schema_dummy import make_output_schema_dummy


def load_schema(schema_name: str) -> dict:
    """Helper to load schemas from the project's /schema directory."""
    # Locates the 'schema' folder at the project root
    project_root = Path(__file__).parent.parent.parent
    schema_path = project_root / "schema" / schema_name
    
    if not schema_path.exists():
        pytest.fail(f"Schema file missing at {schema_path}")
        
    with open(schema_path) as f:
        return json.load(f)


def to_json_safe(obj):
    """
    Recursively converts NumPy types to JSON-serializable Python types.
    Essential for Rule 4 (SSoT) compliance when validating against JSON schemas.
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_safe(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    return obj


class TestSchemaContracts:
    """
    SYSTEM AUDITOR: Verifies that the SSoT helpers generate data 
    that strictly adheres to the JSON schemas.
    """

    def test_input_dummy_matches_schema(self):
        """Validates that the SolverInput dummy matches solver_input_schema.json"""
        schema = load_schema("solver_input_schema.json")
        
        # Create input and convert to dict
        input_obj = create_validated_input(nx=4, ny=4, nz=4)
        payload = input_obj.to_dict()
        
        # Sanitize for JSON Schema (converts numpy arrays/scalars)
        json_safe_payload = to_json_safe(payload)
        
        try:
            jsonschema.validate(instance=json_safe_payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            # We provide the path to the error to make fixing the dummy easier
            error_path = " -> ".join([str(p) for p in e.path])
            pytest.fail(f"Input Contract Violation at [{error_path}]: {e.message}")

    def test_output_dummy_matches_schema(self):
        """Validates that the SolverState dummy matches solver_results_schema.json"""
        schema = load_schema("solver_results_schema.json")
        
        # Generate the Output Dummy (SolverState)
        state = make_output_schema_dummy(nx=4, ny=4, nz=4)
        
        # Transform to dict
        payload = state.to_dict()
        
        # Sanitize
        json_safe_payload = to_json_safe(payload)
        
        try:
            jsonschema.validate(instance=json_safe_payload, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            error_path = " -> ".join([str(p) for p in e.path])
            pytest.fail(f"Output Contract Violation at [{error_path}]: {e.message}")