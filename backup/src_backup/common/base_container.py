# src/common/base_container.py
import json
from collections.abc import Iterator
from typing import Any

import jsonschema
import numpy as np


class ValidatedContainer:
    """The 'Security Guard' logic. Now with memory-efficient slots and O(1) attribute validation."""
    __slots__ = []  
    _ALLOWED_ATTRS = None

    def __iter__(self) -> Iterator[str]:
        """Helper to iterate over attributes defined in slots across the hierarchy."""
        for cls in reversed(self.__class__.__mro__):
            yield from getattr(cls, '__slots__', [])
    
    def _get_safe(self, name: str) -> Any:
        # Rule 5: Explicit or Error. No fallbacks.
        attr_name = f"_{name}"
        if not hasattr(self, attr_name):
            raise AttributeError(f"Coding Error: '{attr_name}' not defined in {self.__class__.__name__}.")
        val = getattr(self, attr_name)
        if val is None:
            raise RuntimeError(f"Access Error: '{name}' in {self.__class__.__name__} is uninitialized.")
        return val

    def _set_safe(self, name: str, value: Any, expected_type: type):
        if value is not None and not isinstance(value, expected_type):
            raise TypeError(f"Validation Error: '{name}' must be {expected_type}, got {type(value)}.")
        setattr(self, f"_{name}", value)

    def validate_against_schema(self, schema_path: str):
        """Final Firewall: Validates current state against the SSoT JSON Schema."""
        with open(schema_path) as f:
            schema = json.load(f)
            
        try:
            data_to_validate = self.to_dict()
            jsonschema.validate(instance=data_to_validate, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            # 1. Isolate the specific sub-dictionary/value that failed
            failed_instance = e.instance
            
            # 2. Extract Type Metadata: Handle NumPy specifically for Scientific Integrity
            if hasattr(failed_instance, "shape"):
                # If it's a NumPy array, identify its dimensions and type
                received_type = f"numpy.ndarray (shape: {failed_instance.shape}, dtype: {failed_instance.dtype})"
            else:
                received_type = type(failed_instance).__name__
            
            # 3. Create a descriptive path
            path_to_error = " -> ".join([str(p) for p in e.path]) if e.path else "root"
            
            # 4. Generate the Diagnostic Report
            print("\n" + "!" * 60)
            print("❌ SCHEMA VALIDATION FAILED")
            print(f"CLASS:      {self.__class__.__name__}")
            print(f"LOCATION:   {path_to_error}")
            print(f"RULE:       {e.validator}: {e.validator_value}")
            print(f"RECEIVED:   {received_type}")
            print(f"DATA TRACE: {repr(failed_instance)[:200]}...")
            print("!" * 60 + "\n")
            
            # 5. Raise with precise intent
            raise ValueError(
                f"\n[Validation Failure] {self.__class__.__name__} at '{path_to_error}': "
                f"Expected {e.validator_value}, but received {received_type}. "
                f"Data fragment: {repr(failed_instance)[:100]}"
            ) from None

    def __setattr__(self, name: str, value: Any):
        if self._ALLOWED_ATTRS is None:
            allowed = set()
            for cls in self.__class__.__mro__:
                allowed.update(getattr(cls, '__slots__', []))
            self.__class__._ALLOWED_ATTRS = frozenset(allowed)
        
        # Check if the name is an allowed slot OR a property descriptor
        is_slot = name in self._ALLOWED_ATTRS
        is_property = isinstance(getattr(self.__class__, name, None), property)
        
        if not (is_slot or is_property):
            raise AttributeError(f"Memory Leak Prevention: '{name}' not in __slots__ for {self.__class__.__name__}")
        
        super().__setattr__(name, value)
    
    def to_dict(self) -> dict:
        """Serializes the container using the slots hierarchy (SSoT compliant)."""
        out = {}
        for attr in self:
            val = getattr(self, attr, None)
            if val is None:
                continue
                
            clean_key = attr.lstrip('_')
            
            # Rule 9: Hybrid Memory Foundation serialization
            if isinstance(val, ValidatedContainer):
                out[clean_key] = val.to_dict()
            elif isinstance(val, np.ndarray):
                out[clean_key] = val.tolist()
            elif hasattr(val, "toarray"):
                out[clean_key] = val.toarray().tolist()
            elif isinstance(val, dict):
                out[clean_key] = {k: (v.toarray().tolist() if hasattr(v, "toarray") 
                        else (v.tolist() if isinstance(v, np.ndarray) else v)) 
                        for k, v in val.items()}
            elif isinstance(val, list):
                out[clean_key] = [(i.to_dict() if isinstance(i, ValidatedContainer) else i) for i in val]
            else:
                out[clean_key] = val
        return out