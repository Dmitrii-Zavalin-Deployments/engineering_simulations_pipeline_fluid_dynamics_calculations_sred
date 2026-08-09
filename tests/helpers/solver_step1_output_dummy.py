# tests/helpers/solver_step1_output_dummy.py

"""
Archivist Testing: Explicit SolverState Hydration.

Compliance:
- Rule 4: SSoT (Hierarchy over convenience).
- Rule 5: Deterministic Initialization (Constructor-based injection).
"""

import numpy as np

from src.common.solver_state import (
    BoundaryCondition,
    BoundaryConditionManager,
    DomainManager,
    ExternalForceManager,
    FieldManager,
    FluidPropertiesManager,
    GridManager,
    InitialConditionManager,
    MaskManager,
    PhysicalConstraintsManager,  # NEW IMPORT
    SimulationParameterManager,
    SolverState,
)


def make_step1_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4) -> SolverState:
    """
    Hydrates a SolverState prototype using atomic constructor injection.
    Validation is triggered immediately upon instantiation of each manager.
    """
    state = SolverState()
    
    # 1. Geometry & Domain: Atomic Constructor Injection
    state._grid = GridManager()
    state._grid._x_min, state._grid._x_max = 0.0, 1.0
    state._grid._y_min, state._grid._y_max = 0.0, 1.0
    state._grid._z_min, state._grid._z_max = 0.0, 1.0
    state._grid._nx, state._grid._ny, state._grid._nz = nx, ny, nz
    
    state._domain_configuration = DomainManager()
    state._domain_configuration._type = "INTERNAL"
    state._domain_configuration._reference_velocity = np.array([0.0, 0.0, 0.0])
    
    # 2. Physics & Foundation: Atomic Constructor Injection
    state._fluid_properties = FluidPropertiesManager()
    state._fluid_properties._density = 1000.0
    state._fluid_properties._viscosity = 0.001
    
    state._initial_conditions = InitialConditionManager()
    state._initial_conditions._velocity = np.array([0.0, 0.0, 0.0])
    state._initial_conditions._pressure = 0.0
    
    state._simulation_parameters = SimulationParameterManager()
    state._simulation_parameters._time_step = 0.001
    state._simulation_parameters._total_time = 1.0
    state._simulation_parameters._output_interval = 1
    
    state._external_forces = ExternalForceManager()
    state._external_forces._force_vector = np.array([0.0, 0.0, -9.81])

    # --- NEW: PHYSICAL CONSTRAINTS INJECTION ---
    state._physical_constraints = PhysicalConstraintsManager()
    state._physical_constraints._min_velocity = -100.0
    state._physical_constraints._max_velocity = 100.0
    state._physical_constraints._min_pressure = -1e6
    state._physical_constraints._max_pressure = 1e6
    # -------------------------------------------

    # 3. Foundation Allocation: Explicit call per Rule 9
    state._fields = FieldManager()
    ghosted_nx, ghosted_ny, ghosted_nz = nx + 2, ny + 2, nz + 2
    state._fields.allocate(ghosted_nx * ghosted_ny * ghosted_nz)
    
    # 4. Topology: Explicit injection
    state._mask = MaskManager()
    state._mask._mask = np.ones((nx, ny, nz), dtype=int)
    
    # 5. Boundary Condition Setup: Atomic instantiation
    state._boundary_conditions = BoundaryConditionManager()
    # Create individual conditions
    conds = []
    for loc, typ, vals in [
        ('x_min', 'inflow', {'u': 1.0, 'v': 0.0, 'w': 0.0}),
        ('x_max', 'outflow', {'p': 0.0}),
        ('y_min', 'no-slip', {'u': 0.0, 'v': 0.0, 'w': 0.0}),
        ('y_max', 'no-slip', {'u': 0.0, 'v': 0.0, 'w': 0.0}),
        ('z_min', 'no-slip', {'u': 0.0, 'v': 0.0, 'w': 0.0}),
        ('z_max', 'no-slip', {'u': 0.0, 'v': 0.0, 'w': 0.0}),
        ('wall', 'no-slip', {'u': 0.0, 'v': 0.0, 'w': 0.0})
    ]:
        bc = BoundaryCondition()
        bc._location = loc
        bc._type = typ
        bc._values = vals
        conds.append(bc)
    
    state._boundary_conditions._conditions = conds
    
    # 6. Engine State: Explicit declaration
    state._iteration = 0
    state._time = 0.0
    state._ready_for_time_loop = False

    return state