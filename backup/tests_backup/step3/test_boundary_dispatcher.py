# tests/step3/test_boundary_dispatcher.py

from unittest.mock import MagicMock

import pytest

from src.step3.boundaries.dispatcher import get_applicable_boundary_configs


def test_get_applicable_configs_external_interior():
    """
    Targets Line 34: EXTERNAL domain with no ghost neighbors (interior cell).
    """
    # 1. Setup an interior block (no ghost neighbors)
    mock_block = MagicMock()
    mock_block.i_minus.is_ghost = False
    mock_block.i_plus.is_ghost = False
    mock_block.j_minus.is_ghost = False
    mock_block.j_plus.is_ghost = False
    mock_block.k_minus.is_ghost = False
    mock_block.k_plus.is_ghost = False
    
    # 2. Setup EXTERNAL domain config
    domain_cfg = {"type": "EXTERNAL", "reference_velocity": [10.0, 0.0, 0.0]}
    
    # 3. Dispatch
    configs = get_applicable_boundary_configs(
        block=mock_block, 
        boundary_cfg=[], 
        grid=None, 
        domain_cfg=domain_cfg
    )
    
    # 4. Assert line 34-38 logic
    assert len(configs) == 1
    assert configs[0]["location"] == "interior"
    assert configs[0]["type"] == "fluid_gas"

def test_get_applicable_configs_external_boundary():
    """
    Targets Lines 27-31: EXTERNAL domain at a physical boundary (x_min).
    """
    mock_block = MagicMock()
    mock_block.i_minus.is_ghost = True # Triggers x_min
    
    domain_cfg = {"type": "EXTERNAL", "reference_velocity": [5.0, 1.0, 0.0]}
    
    configs = get_applicable_boundary_configs(mock_block, [], None, domain_cfg)
    
    assert configs[0]["location"] == "x_min"
    assert configs[0]["type"] == "free-stream"
    assert configs[0]["values"]["u"] == 5.0

def test_get_applicable_configs_external_missing_velocity():
    """
    Targets Lines 23-25: KeyError when reference_velocity is missing.
    """
    mock_block = MagicMock()
    mock_block.i_minus.is_ghost = True
    domain_cfg = {"type": "EXTERNAL"} # Missing key
    
    with pytest.raises(KeyError, match="Missing 'reference_velocity'"):
        get_applicable_boundary_configs(mock_block, [], None, domain_cfg)