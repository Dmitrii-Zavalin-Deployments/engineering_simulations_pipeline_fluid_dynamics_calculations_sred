import logging

import numpy as np

from src.common.base_container import ValidatedContainer
from src.common.field_schema import FI
from src.common.grid_math import get_flat_index

logger = logging.getLogger(__name__)

# =========================================================
# POST: PRE-FLIGHT INTEGRITY CHECK (Rule 9 Sentinel)
# =========================================================

def verify_foundation_integrity(state):
    """
    POST (Power-On Self-Test): Performs a 'Pre-Flight Check' on the memory wiring.
    Rule 9 Sentinel: Uses Identity Priming [Value = Index + (Field_ID / 10.0)].
    Verifies object-pointer mapping and physical memory alignment.
    """
    if state.fields is None or state.fields.data is None:
        logger.error("POST FAILED: Fields buffer not initialized.")
        raise RuntimeError("POST FAILED: Fields buffer not initialized.")
        
    logger.info("POST: Initiating Pre-Flight Memory Integrity Check...")
    
    # 1. Physical Alignment Check (Performance Sentinel)
    # Checks if the buffer is 64-byte aligned for optimal SIMD/Vectorized operations
    alignment = state.fields.data.ctypes.data % 64
    if alignment != 0:
        logger.warning(f"POST ALIGNMENT: Buffer is not 64-byte aligned (Offset: {alignment}). Efficiency may drop.")
    else:
        logger.debug("POST ALIGNMENT: 64-byte memory boundary verified.")

    # 2. Architecture Bounds Check
    num_cells = state.fields.data.shape[0]
    if len(state.stencil_matrix) > num_cells:
        logger.error(f"POST OVERFLOW: Stencil count ({len(state.stencil_matrix)}) > Buffer size ({num_cells})")
        raise RuntimeError("POST OVERFLOW: Architecture integrity compromised.")
        
    # 3. Identity Priming (The "Memory Tracker")
    # Store original state to restore after the test
    original_data = state.fields.data.copy()
    
    try:
        # Prime EVERY field in the monolithic buffer with a unique coordinate-based ID
        for f in FI:
            state.fields.data[:, f] = np.arange(num_cells, dtype=float) + (float(f) / 10.0)

        # 4. Pointer Validation Loop
        for idx, block in enumerate(state.stencil_matrix):
            c = block.center
            if c.is_ghost:
                continue
            
            # Verify P-field pointer
            expected_p = float(c.index) + (float(FI.P) / 10.0)
            if not np.isclose(c.p, expected_p):
                logger.critical(f"MEMORY DRIFT [P]: Index {idx} | Expected {expected_p}, Got {c.p}")
                raise RuntimeError(f"CRITICAL: Memory Swap at Stencil Index {idx}!")

            # Verify Velocity-field pointers (X-component check)
            expected_vx = float(c.index) + (float(FI.VX) / 10.0)
            if not np.isclose(c.u[0], expected_vx):
                logger.critical(f"MEMORY DRIFT [VX]: Index {idx} | Expected {expected_vx}, Got {c.u[0]}")
                raise RuntimeError(f"CRITICAL: Vector Component Displacement at Index {idx}!")
        
        logger.info("POST: Memory integrity verified. All pointers aligned to Monolithic Foundation.")

    finally:
        # Restore the actual simulation data
        state.fields.data[:] = original_data

# =========================================================
# THE DEPARTMENT SAFES (Memory-Hardened Managers)
# =========================================================

class PhysicalConstraintsManager(ValidatedContainer):
    __slots__ = ['_max_pressure', '_max_velocity', '_min_pressure', '_min_velocity']
    
    def __init__(self):
        self._min_velocity = self._max_velocity = None
        self._min_pressure = self._max_pressure = None

    def to_dict(self):
        return {
            "min_velocity": self.min_velocity,
            "max_velocity": self.max_velocity,
            "min_pressure": self.min_pressure,
            "max_pressure": self.max_pressure
        }

    @property
    def min_velocity(self) -> float: return self._get_safe("min_velocity")
    @min_velocity.setter
    def min_velocity(self, v: float): self._set_safe("min_velocity", v, (float, int))

    @property
    def max_velocity(self) -> float: return self._get_safe("max_velocity")
    @max_velocity.setter
    def max_velocity(self, v: float): self._set_safe("max_velocity", v, (float, int))

    @property
    def min_pressure(self) -> float: return self._get_safe("min_pressure")
    @min_pressure.setter
    def min_pressure(self, v: float): self._set_safe("min_pressure", v, (float, int))

    @property
    def max_pressure(self) -> float: return self._get_safe("max_pressure")
    @max_pressure.setter
    def max_pressure(self, v: float): self._set_safe("max_pressure", v, (float, int))

class DomainManager(ValidatedContainer):
    __slots__ = ['_reference_velocity', '_type']
    
    def __init__(self):
        self._type = None
        self._reference_velocity = None

    @property
    def type(self) -> str: return self._get_safe("type")
    @type.setter
    def type(self, value: str):
        if value not in ["INTERNAL", "EXTERNAL"]:
            raise ValueError(f"Invalid domain type: {value}. Must be INTERNAL or EXTERNAL.")
        self._set_safe("type", value, str)

    @property
    def reference_velocity(self) -> np.ndarray: return self._get_safe("reference_velocity")
    @reference_velocity.setter
    def reference_velocity(self, value: np.ndarray):
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise TypeError("reference_velocity must be a 3D NumPy array.")
        self._set_safe("reference_velocity", value, np.ndarray)

class GridManager(ValidatedContainer):
    __slots__ = ['_nx', '_ny', '_nz', '_x_max', '_x_min', '_y_max', '_y_min', '_z_max', '_z_min']
    
    def __init__(self):
        self._x_min = self._x_max = self._y_min = self._y_max = self._z_min = self._z_max = None
        self._nx = self._ny = self._nz = None

    @property
    def x_min(self) -> float: return self._get_safe("x_min")
    @x_min.setter
    def x_min(self, value: float): self._set_safe("x_min", value, float)
    
    @property
    def x_max(self) -> float: return self._get_safe("x_max")
    @x_max.setter
    def x_max(self, value: float): self._set_safe("x_max", value, float)

    @property
    def y_min(self) -> float: return self._get_safe("y_min")
    @y_min.setter
    def y_min(self, value: float): self._set_safe("y_min", value, float)

    @property
    def y_max(self) -> float: return self._get_safe("y_max")
    @y_max.setter
    def y_max(self, value: float): self._set_safe("y_max", value, float)

    @property
    def z_min(self) -> float: return self._get_safe("z_min")
    @z_min.setter
    def z_min(self, value: float): self._set_safe("z_min", value, float)

    @property
    def z_max(self) -> float: return self._get_safe("z_max")
    @z_max.setter
    def z_max(self, value: float): self._set_safe("z_max", value, float)

    @property
    def nx(self) -> int: return self._get_safe("nx")
    @nx.setter
    def nx(self, value: int):
        if value < 1: raise ValueError("nx must be >= 1")
        self._set_safe("nx", value, int)

    @property
    def ny(self) -> int: return self._get_safe("ny")
    @ny.setter
    def ny(self, value: int):
        if value < 1: raise ValueError("ny must be >= 1")
        self._set_safe("ny", value, int)

    @property
    def nz(self) -> int: return self._get_safe("nz")
    @nz.setter
    def nz(self, value: int):
        if value < 1: raise ValueError("nz must be >= 1")
        self._set_safe("nz", value, int)

    @property
    def dx(self) -> float: return (self.x_max - self.x_min) / self.nx
    @property
    def dy(self) -> float: return (self.y_max - self.y_min) / self.ny
    @property
    def dz(self) -> float: return (self.z_max - self.z_min) / self.nz

class FluidPropertiesManager(ValidatedContainer):
    __slots__ = ['_density', '_viscosity']
    
    def __init__(self):
        self._density = self._viscosity = None

    @property
    def density(self) -> float: return self._get_safe("density")
    @density.setter
    def density(self, value: float):
        if value is not None and value <= 0: raise ValueError(f"Density must be > 0, got {value}.")
        self._set_safe("density", value, float)

    @property
    def viscosity(self) -> float: return self._get_safe("viscosity")
    @viscosity.setter
    def viscosity(self, value: float):
        if value is not None and value < 0: raise ValueError(f"Viscosity must be >= 0, got {value}.")
        self._set_safe("viscosity", value, float)

class InitialConditionManager(ValidatedContainer):
    __slots__ = ["_pressure", "_velocity"]

    def to_dict(self):
        return {"velocity": self.velocity.tolist(), "pressure": self.pressure}

    def __init__(self):
        self._velocity = self._pressure = None

    @property
    def velocity(self) -> np.ndarray: return self._get_safe("velocity")
    @velocity.setter
    def velocity(self, value: np.ndarray):
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise ValueError("Velocity must be a 3D NumPy array.")
        self._set_safe("velocity", value, np.ndarray)

    @property
    def pressure(self) -> float: return self._get_safe("pressure")
    @pressure.setter
    def pressure(self, value: float): self._set_safe("pressure", value, (float, int))

class SimulationParameterManager(ValidatedContainer):
    __slots__ = ['_output_interval', '_time_step', '_total_time']
    
    def __init__(self):
        self._time_step = self._total_time = self._output_interval = None

    @property
    def time_step(self) -> float: return self._get_safe("time_step")
    @time_step.setter
    def time_step(self, value: float):
        if value is not None and value <= 0: raise ValueError(f"time_step must be > 0, got {value}.")
        self._set_safe("time_step", value, (float, int))

    @property
    def total_time(self) -> float: return self._get_safe("total_time")
    @total_time.setter
    def total_time(self, value: float):
        if value is not None and value <= 0: raise ValueError(f"total_time must be > 0, got {value}.")
        self._set_safe("total_time", value, (float, int))

    @property
    def output_interval(self) -> int: return self._get_safe("output_interval")
    @output_interval.setter
    def output_interval(self, value: int):
        if value is not None and value < 1: raise ValueError(f"output_interval must be >= 1, got {value}.")
        self._set_safe("output_interval", value, int)

class BoundaryCondition(ValidatedContainer):
    __slots__ = ['_location', '_type', '_values']
    
    def __init__(self):
        self._location = self._type = self._values = None

    @property
    def location(self) -> str: return self._get_safe("location")
    @location.setter
    def location(self, value: str):
        valid = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"]
        if value is not None and value not in valid: raise ValueError(f"Invalid location '{value}'.")
        self._set_safe("location", value, str)

    @property
    def type(self) -> str: return self._get_safe("type")
    @type.setter
    def type(self, value: str):
        valid = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if value is not None and value not in valid: raise ValueError(f"Invalid type '{value}'.")
        self._set_safe("type", value, str)

    @property
    def values(self) -> dict: return self._get_safe("values")
    @values.setter
    def values(self, value: dict):
        if value is not None and not isinstance(value, dict): raise TypeError("values must be a dict.")
        self._set_safe("values", value, dict)

class BoundaryConditionManager(ValidatedContainer):
    __slots__ = ["_conditions"]

    def to_dict(self):
        return [c.to_dict() for c in self.conditions]

    def __init__(self):
        self._conditions = None

    @property
    def conditions(self) -> list[BoundaryCondition]: return self._get_safe("conditions")
    @conditions.setter
    def conditions(self, value: list[BoundaryCondition]):
        if value is not None and not isinstance(value, list): raise TypeError("Must be a list.")
        self._set_safe("conditions", value, list)

    def add_condition(self, condition: BoundaryCondition):
        current = self._conditions if self._conditions is not None else []
        if not isinstance(condition, BoundaryCondition): raise TypeError("Must be BoundaryCondition.")
        current.append(condition)
        self.conditions = current

class MaskManager(ValidatedContainer):
    __slots__ = ['_mask']
    
    def to_dict(self):
        if self._mask is None:
            raise RuntimeError("MaskManager: _mask is uninitialized. Cannot serialize.")
        return self._mask.flatten().tolist()
    
    def __init__(self):
        self._mask = None

    @property
    def mask(self) -> np.ndarray: return self._get_safe("mask")
    @mask.setter
    def mask(self, value: np.ndarray):
        if value is not None and (
            not isinstance(value, np.ndarray) or not np.all(np.isin(value, [-1, 0, 1]))
        ):
                raise ValueError("Mask must be a NumPy array of -1, 0, 1.")
        self._set_safe("mask", value, np.ndarray)

class ExternalForceManager(ValidatedContainer):
    def to_dict(self):
        if self.force_vector is None:
            raise AttributeError("ExternalForceManager: force_vector must be initialized.")
        return {"force_vector": self.force_vector.tolist()}
    __slots__ = ['_force_vector']
    
    def __init__(self):
        self._force_vector = None

    @property
    def force_vector(self) -> np.ndarray: return self._get_safe("force_vector")
    @force_vector.setter
    def force_vector(self, value: np.ndarray):
        if value is not None and (not isinstance(value, np.ndarray) or value.size != 3):
            raise ValueError("force_vector must be a 3D NumPy array.")
        self._set_safe("force_vector", value, np.ndarray)

class FieldManager(ValidatedContainer):
    __slots__ = ['_data']
    
    def __init__(self):
        self._data = None

    @property
    def data(self) -> np.ndarray: return self._get_safe("data")
    @data.setter
    def data(self, value: np.ndarray):
        if not isinstance(value, np.ndarray): raise TypeError("Field data must be a NumPy array.")
        self._set_safe("data", value, np.ndarray)

    def allocate(self, n_cells: int, dtype=np.float64):
        self._data = np.empty((n_cells, FI.num_fields()), dtype=dtype); self._data[:] = 0.0

class ManifestManager(ValidatedContainer):
    __slots__ = ['_output_directory', '_saved_snapshots']
    
    def __init__(self):
        self._saved_snapshots = []
        self._output_directory = "output"

    @property
    def saved_snapshots(self) -> list: return self._get_safe("saved_snapshots")
    @saved_snapshots.setter
    def saved_snapshots(self, value: list): self._set_safe("saved_snapshots", value, list)

    @property
    def output_directory(self) -> str: return self._get_safe("output_directory")
    @output_directory.setter
    def output_directory(self, value: str): self._set_safe("output_directory", value, str)

# =========================================================
# THE UNIVERSAL CONTAINER (The Constitution)
# =========================================================

class SolverState(ValidatedContainer):
    __slots__ = [
        '_boundary_conditions',
        '_cache_buffer',
        '_domain_configuration',
        '_external_forces',
        '_fields',
        '_fluid_properties',
        '_grid',
        '_initial_conditions',
        '_iteration',
        '_manifest',
        '_mask',
        '_physical_constraints',
        '_ready_for_time_loop',
        '_simulation_parameters',
        '_stencil_matrix',
        '_time'
    ]

    def __init__(self):
        super().__init__()
        self._physical_constraints = None
        self._domain_configuration = self._grid = self._fluid_properties = self._initial_conditions = None
        self._boundary_conditions = self._external_forces = self._simulation_parameters = None
        self._mask = self._fields = self._stencil_matrix = None
        self._iteration = 0
        self._time = 0.0
        self._ready_for_time_loop = False
        self._manifest = ManifestManager() 
        self._cache_buffer = None # Lazy allocation

    @property
    def physical_constraints(self) -> PhysicalConstraintsManager: 
        return self._get_safe("physical_constraints")
    
    @physical_constraints.setter
    def physical_constraints(self, value: PhysicalConstraintsManager): 
        self._set_safe("physical_constraints", value, PhysicalConstraintsManager)

    @property
    def manifest(self) -> ManifestManager: return self._get_safe("manifest")
    @manifest.setter
    def manifest(self, value: ManifestManager): self._set_safe("manifest", value, ManifestManager)

    @property
    def domain_configuration(self) -> DomainManager: return self._get_safe("domain_configuration")
    @domain_configuration.setter
    def domain_configuration(self, value: DomainManager): self._set_safe("domain_configuration", value, DomainManager)

    @property
    def grid(self) -> GridManager: return self._get_safe("grid")
    @grid.setter
    def grid(self, value: GridManager): self._set_safe("grid", value, GridManager)

    @property
    def fluid_properties(self) -> FluidPropertiesManager: return self._get_safe("fluid_properties")
    @fluid_properties.setter
    def fluid_properties(self, value: FluidPropertiesManager): self._set_safe("fluid_properties", value, FluidPropertiesManager)

    @property
    def initial_conditions(self) -> InitialConditionManager: return self._get_safe("initial_conditions")
    @initial_conditions.setter
    def initial_conditions(self, value: InitialConditionManager): self._set_safe("initial_conditions", value, InitialConditionManager)

    @property
    def boundary_conditions(self) -> BoundaryConditionManager: return self._get_safe("boundary_conditions")
    @boundary_conditions.setter
    def boundary_conditions(self, value: BoundaryConditionManager): self._set_safe("boundary_conditions", value, BoundaryConditionManager)

    @property
    def external_forces(self) -> ExternalForceManager: return self._get_safe("external_forces")
    @external_forces.setter
    def external_forces(self, value: ExternalForceManager): self._set_safe("external_forces", value, ExternalForceManager)

    @property
    def simulation_parameters(self) -> SimulationParameterManager: return self._get_safe("simulation_parameters")
    @simulation_parameters.setter
    def simulation_parameters(self, value: SimulationParameterManager): self._set_safe("simulation_parameters", value, SimulationParameterManager)

    @property
    def mask(self) -> MaskManager: return self._get_safe("mask")
    @mask.setter
    def mask(self, value: MaskManager): self._set_safe("mask", value, MaskManager)

    @property
    def fields(self) -> FieldManager: return self._get_safe("fields")
    @fields.setter
    def fields(self, value: FieldManager): self._set_safe("fields", value, FieldManager)

    @property
    def stencil_matrix(self) -> list: return self._get_safe("stencil_matrix")
    @stencil_matrix.setter
    def stencil_matrix(self, value: list): self._set_safe("stencil_matrix", value, list)

    @property
    def iteration(self) -> int: return self._iteration
    @iteration.setter
    def iteration(self, value: int): self._iteration = value

    @property
    def time(self) -> float: return self._time
    @time.setter
    def time(self, value: float): self._time = value

    # =========================================================
    # RULE 9: MEMORY RECOVERY (Anti-Frankenstein Protocol)
    # =========================================================

    def capture_stable_state(self):
        """Rule 9: Deep-copying the foundation to prevent pointer aliasing."""
        if self._cache_buffer is None:
            self._cache_buffer = np.zeros_like(self.fields.data)
            logger.info("CACHE: Rollback buffer allocated.")
        
        # FIX: Use np.copyto or .copy() to ensure data is physically moved in RAM
        np.copyto(self._cache_buffer, self.fields.data)
        logger.debug("CACHE: Stable state captured.")

    def rollback_to_stable_state(self):
        """
        Reverts the fields buffer to the last stable snapshot.
        Wipes the 'numerical storm' left by a failed attempt.
        """
        if self._cache_buffer is None:
            raise RuntimeError("CRITICAL: Rollback requested but no cache exists.")
        
        self.fields.data[:] = self._cache_buffer[:]
        logger.warning(f"ROLLBACK: Memory reverted to state at start of Iteration {self.iteration}.")

    # =========================================================
    # RULE 7: VECTORIZED PHYSICAL AUDIT
    # =========================================================
    def audit_physical_bounds(self):
        """
        Rule 7: Vectorized Physical Audit with Forensic Logging.
        Strictly enforces JSON Schema constraints: [min/max velocity, min/max pressure].
        """
        pc = self.physical_constraints
        fields = self.fields.data 
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        logger.debug(f"AUDIT [Start]: Iteration {self.iteration}")
        
        # 1. Finite Status & Velocity Check
        v_indices = [FI.VX, FI.VY, FI.VZ, FI.VX_STAR, FI.VY_STAR, FI.VZ_STAR]
        v_fields = fields[:, v_indices]
        
        if not np.isfinite(v_fields).all():
            logger.error("AUDIT [Explosion]: NaN/Inf detected in velocity fields.")
            raise ArithmeticError("NUMERICAL EXPLOSION: NaN/Inf detected.")

        # --- [New]: Min/Max Velocity Enforcement ---
        v_min_observed = np.min(v_fields)
        v_max_observed = np.max(v_fields)

        if v_max_observed > pc.max_velocity or v_min_observed < pc.min_velocity:
            logger.error(
                f"AUDIT [Limit]: Velocity range [{v_min_observed:.4e}, {v_max_observed:.4e}] "
                f"exceeds physical bounds [{pc.min_velocity:.4e}, {pc.max_velocity:.4e}]."
            )
            raise ArithmeticError("STABILITY TRIGGER: Velocity Corridor Violation.")

        # 2. Real Physical Pressure Reconstruction
        
        # 2.1 Find reference boundary (Search for Dirichlet 'p' anchor)
        ref_bc = next((bc for bc in self.boundary_conditions.conditions 
                    if bc.type in ("pressure", "outflow", "inflow") and "p" in bc.values), None)

        if ref_bc is None:
            raise RuntimeError("No pressure reference boundary found (Required for Rule 7).")

        # 2.2 Vectorized Index Extraction (Geometric Slicing via SSoT)
        # Generates the coordinate map using the central grid_math logic
        all_indices = np.fromfunction(
            lambda k, j, i: get_flat_index(i, j, k, nx, ny),
            (nz, ny, nx),
            dtype=int
        )
        
        loc = ref_bc.location
        if loc == "x_min":   ref_indices = all_indices[:, :, 0]
        elif loc == "x_max": ref_indices = all_indices[:, :, -1]
        elif loc == "y_min": ref_indices = all_indices[:, 0, :]
        elif loc == "y_max": ref_indices = all_indices[:, -1, :]
        elif loc == "z_min": ref_indices = all_indices[0, :, :]
        elif loc == "z_max": ref_indices = all_indices[-1, :, :]
        else:
            raise RuntimeError(f"Unsupported boundary location '{loc}'")

        # 2.3 Pressure Grounding
        p_trial = fields[:, FI.P_NEXT]
        p_ref_value = np.mean(p_trial[ref_indices.ravel()])
        p_ref_physical = ref_bc.values["p"]
        
        p_real = p_trial - p_ref_value + p_ref_physical

        # 3. Pressure Range Audit
        p_min, p_max = p_real.min(), p_real.max()
        
        if not (np.isfinite(p_min) and np.isfinite(p_max)):
            logger.error("AUDIT [Explosion]: NaN/Inf detected in pressure field.")
            raise ArithmeticError("NUMERICAL EXPLOSION: Pressure NaN/Inf detected.")

        if p_min < pc.min_pressure or p_max > pc.max_pressure:
            logger.error(
                f"AUDIT [Explosion]: Real pressure [{p_min:.2e}, {p_max:.2e}] "
                f"out of bounds [{pc.min_pressure:.2e}, {pc.max_pressure:.2e}]."
            )
            raise ArithmeticError("STABILITY TRIGGER: Pressure Corridor Violation.")

        logger.debug(
            f"AUDIT [Success]: V_range: [{v_min_observed:.2e}, {v_max_observed:.2e}], "
            f"P_range: [{p_min:.2e}, {p_max:.2e}]"
        )

    def validate_physical_readiness(self):
        if self.fields is None or self.fields.data is None:
            raise RuntimeError("CRITICAL: Foundation buffer is missing.")
        
        # Enforce that constraints are actually set before starting
        if self.physical_constraints is None:
             raise RuntimeError("CRITICAL: Physical Constraints are not defined.")

        if np.any(np.isnan(self.fields.data)) or np.any(np.isinf(self.fields.data)):
            raise RuntimeError("CRITICAL: NaNs/Infs detected in Foundation buffer!")
        if self.grid.nx is None or self.grid.nx < 1:
            raise RuntimeError("CRITICAL: Grid not properly initialized.")

    @property
    def ready_for_time_loop(self) -> bool: return self._ready_for_time_loop

    @ready_for_time_loop.setter
    def ready_for_time_loop(self, value: bool):
        if not isinstance(value, bool): raise TypeError("Must be boolean.")
        if value is True:
            verify_foundation_integrity(self)
            self.validate_physical_readiness()
        self._ready_for_time_loop = value

    def to_dict(self):
        """
        Serializes the solver state by delegating to sub-container methods.
        Ensures SSoT compliance by preventing manual attribute mapping here.
        """
        return {
            "domain_configuration": self.domain_configuration.to_dict(),
            "grid": self.grid.to_dict(),
            "fluid_properties": self.fluid_properties.to_dict(),
            "initial_conditions": self.initial_conditions.to_dict(),
            "simulation_parameters": self.simulation_parameters.to_dict(),
            "physical_constraints": self.physical_constraints.to_dict(),
            "boundary_conditions": self.boundary_conditions.to_dict(),
            "mask": self.mask.to_dict(),
            "external_forces": self.external_forces.to_dict(),
            "manifest": self.manifest.to_dict()
        }
