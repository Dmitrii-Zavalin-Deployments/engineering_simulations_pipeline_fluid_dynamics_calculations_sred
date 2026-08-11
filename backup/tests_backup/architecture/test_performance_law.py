# tests/architecture/test_performance_law.py

import numpy as np

from src.common.cell import Cell
from src.common.stencil_block import StencilBlock
from src.step2.stencil_assembler import assemble_stencil_matrix


def test_law_of_performance_slots():
    """
    Constitutional Rule: __slots__ is Mandatory.
    Verification: Ensure core logic objects (Cell, StencilBlock, CellRegistry) 
    do not have a __dict__ to minimize memory footprint.
    """
    # 1. Verify Cell compliance
    cell = Cell(index=0, fields_buffer=np.zeros((1, 10)), nx_buf=10, ny_buf=10)
    assert not hasattr(cell, '__dict__'), "Phase C Breach: Cell class missing __slots__!"
    assert hasattr(cell, '__slots__'), "Phase C Breach: Cell missing __slots__ definition!"

    # 2. Verify StencilBlock compliance
    # We check the class attributes directly to avoid complex object instantiation
    assert hasattr(StencilBlock, '__slots__'), "Phase C Breach: StencilBlock missing __slots__!"
    assert '_center' in StencilBlock.__slots__, "StencilBlock slots must include topological wiring pointers."

    # 3. Verify CellRegistry compliance (from stencil_assembler.py)
    from src.step2.stencil_assembler import CellRegistry
    assert hasattr(CellRegistry, '__slots__'), "Phase C Breach: CellRegistry missing __slots__!"

def test_law_of_performance_no_object_arrays():
    """
    Constitutional Rule: No Object Arrays (NumPy dtype=object is prohibited).
    Verification: Ensure the local_stencil_list and CellRegistry._cache use 
    standard Python lists for pointer-integrity.
    """
    from src.step2.stencil_assembler import CellRegistry
    
    # Verify that the Flyweight cache in CellRegistry is a standard list, not a NumPy array
    registry = CellRegistry(nx=2, ny=2, nz=2)
    assert isinstance(registry._cache, list), "Rule 0 Breach: CellRegistry cache must be a Python list."
    
    # Verify the return type of the primary assembly function
    # Note: We check the expected behavior of the function signature
    assert assemble_stencil_matrix.__annotations__.get('return') is list, \
        "Rule 0 Breach: assemble_stencil_matrix must return a standard list for O(N) traversal."

def test_hybrid_memory_pattern_integrity():
    """
    Constitutional Rule: Hybrid Memory Pattern (Logic for Objects, Math for Arrays).
    Verification: Cell objects must return native floats for fields, not NumPy views.
    """
    buffer = np.zeros((1, 10))
    buffer[0, 0] = 1.23456789
    cell = Cell(index=0, fields_buffer=buffer, nx_buf=1, ny_buf=1)
    
    val = cell.get_field(0)
    assert isinstance(val, float), f"Rule 9 Breach: Cell.get_field returned {type(val)} instead of native float."
    assert not isinstance(val, np.ndarray), "Rule 9 Breach: NumPy leakage detected in logic layer."