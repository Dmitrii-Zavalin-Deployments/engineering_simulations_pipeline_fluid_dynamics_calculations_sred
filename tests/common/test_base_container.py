# tests/common/test_base_container.py


import numpy as np
import pytest

from src.common.base_container import ValidatedContainer


# --- Setup Dummy Classes for Testing ---
class MockContainer(ValidatedContainer):
    __slots__ = ['_density', '_velocity']
    def __init__(self):
        self._velocity = None
        self._density = 1.0

class SparseMock:
    """Simulates a SciPy sparse matrix for Line 106 coverage."""
    def toarray(self):
        return np.array([[1, 0], [0, 1]])

# --- Test Cases ---

def test_get_safe_undefined_attribute_error():
    """Coverage for Line 24: AttributeError when attribute is not in slots."""
    container = MockContainer()
    with pytest.raises(AttributeError, match="not defined in MockContainer"):
        # Accessing 'pressure' which is NOT in __slots__
        container._get_safe("pressure")

def test_set_safe_type_error():
    """Coverage for Line 32: TypeError on invalid type assignment."""
    container = MockContainer()
    with pytest.raises(TypeError, match="must be <class 'float'>"):
        # Expected float, providing string
        container._set_safe("density", "heavy", float)

def test_numpy_schema_validation_error(tmp_path, mocker):
    """
    Coverage for Line 50: Explicitly trigger the NumPy diagnostic.
    Forces the validator to receive a raw array, hitting the forensic metadata block.
    """
    # 1. Setup Schema (Strict integer requirement)
    schema = {
        "type": "object",
        "properties": {"velocity": {"type": "integer"}},
        "required": ["velocity"]
    }
    schema_file = tmp_path / "schema.json"
    import json
    schema_file.write_text(json.dumps(schema))

    # 2. Setup Container with scientific data
    container = MockContainer()
    raw_array = np.array([1.1, 2.2])
    container._velocity = raw_array

    # 3. THE FIX: Patch 'to_dict' on the MockContainer class directly.
    # We return the raw array inside the dict so Line 48 'hasattr(val, "shape")' is True.
    mocker.patch.object(
        MockContainer, 
        'to_dict', 
        return_value={"velocity": raw_array}
    )

    # 4. EXECUTION
    with pytest.raises(ValueError) as exc_info:
        container.validate_against_schema(str(schema_file))
    
    # 5. VERIFICATION: Ensure Line 50's specific forensic output is present
    error_msg = str(exc_info.value)
    assert "numpy.ndarray" in error_msg
    assert "shape: (2,)" in error_msg
    assert "float64" in error_msg

def test_to_dict_with_sparse_toarray():
    """Coverage for Line 106: Serialization of objects with .toarray()."""
    container = MockContainer()
    container._velocity = SparseMock()
    
    result = container.to_dict()
    
    # Verify it converted the sparse-mock to a nested list
    assert result["velocity"] == [[1, 0], [0, 1]]