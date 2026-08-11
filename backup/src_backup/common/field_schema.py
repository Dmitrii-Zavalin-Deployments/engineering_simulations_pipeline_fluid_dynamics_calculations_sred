# src/common/field_schema.py

from enum import IntEnum


class FI(IntEnum):
    """
    FieldIndex (FI) Schema:
    The Single Source of Truth (SSoT) for the global fields_buffer 
    column mapping.
    
    This Enum MUST be used by the FieldManager for memory allocation
    and by Cell objects for pointer-based access to ensure consistency.
    """
    # Primary Velocity Fields
    VX = 0
    VY = 1
    VZ = 2
    
    # Intermediate Predictor Fields
    VX_STAR = 3
    VY_STAR = 4
    VZ_STAR = 5
    
    # Pressure Fields
    P = 6
    P_NEXT = 7
    
    # Topological Mask (Unified into Foundation per Rule 9)
    MASK = 8

    @classmethod
    def num_fields(cls) -> int:
        """
        Returns the total number of fields in the buffer.
        Used by the FieldManager to allocate the Foundation buffer.
        """
        return len(cls)