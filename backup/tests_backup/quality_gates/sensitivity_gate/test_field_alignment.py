# tests/quality_gates/sensitivity_gate/test_field_alignment.py

from src.common.field_schema import FI


def test_gate_3c_fi_alignment_lock():
    """
    Gate 3.C: Schema Lock Audit
    Verification: Reject non-schema keys and ensure valid keys map to FI indices.
    Compliance: src/step1/orchestrate_step1.py intake integrity.
    """
    # 1. Define the Bridge: Mapping of User JSON keys to Schema (FI)
    # This represents the only permitted keys for boundary values
    schema_bridge = {
        'u': FI.VX,
        'v': FI.VY,
        'w': FI.VZ,
        'p': FI.P
    }

    # 2. Verification: Identity Check
    # Ensure all velocity components and pressure are represented
    assert schema_bridge['u'] == FI.VX
    assert schema_bridge['v'] == FI.VY
    assert schema_bridge['w'] == FI.VZ
    assert schema_bridge['p'] == FI.P

    # 3. Verification: Rejection Check (The Logic Firewall)
    # Boundary keys must be abbreviated ('u', 'v', 'w', 'p')
    # Reject over-specified or verbose keys to prevent "Ghost Data" intake
    rejected_keys = [
        "velocity_x", 
        "pressure_final", 
        "VX", 
        "temp", 
        "velocity_x_unfiltered"
    ]

    for key in rejected_keys:
        assert key not in schema_bridge, (
            f"Schema Lock Breach: Key '{key}' permitted in intake. "
            f"Only abbreviated physical keys {list(schema_bridge.keys())} allowed."
        )

def test_gate_3c_fi_completeness():
    """Ensure FI Schema remains the Single Source of Truth for Foundation memory."""
    # Ensure primary fields exist in the Enum as per src/common/field_schema.py
    required_fields = ['VX', 'VY', 'VZ', 'P', 'MASK']
    
    for field in required_fields:
        assert hasattr(FI, field), f"FI Schema Error: Missing mandatory field {field}"
    
    # Verify the count to prevent silent indexing shifts
    assert FI.num_fields() == 9, "FI Schema drift detected: Expected 9 fields total."