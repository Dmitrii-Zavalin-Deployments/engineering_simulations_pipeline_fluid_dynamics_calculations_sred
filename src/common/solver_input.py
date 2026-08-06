from dataclasses import dataclass

from src.common.base_container import ValidatedContainer

# =========================================================
# 1. SUB-DEPARTMENT CONTAINERS
# =========================================================

@dataclass
class PhysicalConstraintsInput(ValidatedContainer):
    __slots__ = ['_max_pressure', '_max_velocity', '_min_pressure', '_min_velocity']
    
    def __init__(self):
        for slot in self.__slots__: setattr(self, slot, None)

    @property
    def min_velocity(self) -> float: return self._get_safe("min_velocity")
    @min_velocity.setter
    def min_velocity(self, v: float): self._set_safe("min_velocity", v, float)

    @property
    def max_velocity(self) -> float: return self._get_safe("max_velocity")
    @max_velocity.setter
    def max_velocity(self, v: float): self._set_safe("max_velocity", v, float)

    @property
    def min_pressure(self) -> float: return self._get_safe("min_pressure")
    @min_pressure.setter
    def min_pressure(self, v: float): self._set_safe("min_pressure", v, float)

    @property
    def max_pressure(self) -> float: return self._get_safe("max_pressure")
    @max_pressure.setter
    def max_pressure(self, v: float): self._set_safe("max_pressure", v, float)

@dataclass
class DomainConfigInput(ValidatedContainer):
    __slots__ = ['_reference_velocity', '_type']
    
    def __init__(self):
        self._type = None
        self._reference_velocity = None

    @property
    def type(self) -> str: return self._get_safe("type")
    @type.setter
    def type(self, v: str):
        if v not in ["INTERNAL", "EXTERNAL"]: 
            raise ValueError(f"Invalid domain type: {v}. Must be INTERNAL or EXTERNAL.")
        self._set_safe("type", v, str)

    @property
    def reference_velocity(self) -> list: return self._get_safe("reference_velocity")
    @reference_velocity.setter
    def reference_velocity(self, v: list):
        if v is not None and len(v) != 3: 
            raise ValueError("reference_velocity must have 3 items [u, v, w]")
        self._set_safe("reference_velocity", v, list)

@dataclass
class GridInput(ValidatedContainer):
    __slots__ = ['_nx', '_ny', '_nz', '_x_max', '_x_min', '_y_max', '_y_min', '_z_max', '_z_min']
    
    def __init__(self):
        for slot in self.__slots__: setattr(self, slot, None)
    
    @property
    def x_min(self) -> float: return self._get_safe("x_min")
    @x_min.setter
    def x_min(self, v: float): self._set_safe("x_min", v, float)
    @property
    def x_max(self) -> float: return self._get_safe("x_max")
    @x_max.setter
    def x_max(self, v: float): self._set_safe("x_max", v, float)
    @property
    def nx(self) -> int: return self._get_safe("nx")
    @nx.setter
    def nx(self, v: int): 
        if v < 1: raise ValueError(f"nx must be >= 1, got {v}")
        self._set_safe("nx", v, int)
    @property
    def y_min(self) -> float: return self._get_safe("y_min")
    @y_min.setter
    def y_min(self, v: float): self._set_safe("y_min", v, float)
    @property
    def y_max(self) -> float: return self._get_safe("y_max")
    @y_max.setter
    def y_max(self, v: float): self._set_safe("y_max", v, float)
    @property
    def ny(self) -> int: return self._get_safe("ny")
    @ny.setter
    def ny(self, v: int): 
        if v < 1: raise ValueError(f"ny must be >= 1, got {v}")
        self._set_safe("ny", v, int)
    @property
    def z_min(self) -> float: return self._get_safe("z_min")
    @z_min.setter
    def z_min(self, v: float): self._set_safe("z_min", v, float)
    @property
    def z_max(self) -> float: return self._get_safe("z_max")
    @z_max.setter
    def z_max(self, v: float): self._set_safe("z_max", v, float)
    @property
    def nz(self) -> int: return self._get_safe("nz")
    @nz.setter
    def nz(self, v: int): 
        if v < 1: raise ValueError(f"nz must be >= 1, got {v}")
        self._set_safe("nz", v, int)

@dataclass
class FluidInput(ValidatedContainer):
    __slots__ = ['_density', '_viscosity']
    
    def __init__(self):
        self._density = None
        self._viscosity = None
        
    @property
    def density(self) -> float: return self._get_safe("density")
    @density.setter
    def density(self, v: float):
        if v <= 0: raise ValueError(f"Density must be > 0, got {v}")
        self._set_safe("density", v, float)
    @property
    def viscosity(self) -> float: return self._get_safe("viscosity")
    @viscosity.setter
    def viscosity(self, v: float):
        if v < 0: raise ValueError(f"Viscosity must be >= 0, got {v}")
        self._set_safe("viscosity", v, float)

@dataclass
class InitialConditionsInput(ValidatedContainer):
    __slots__ = ['_pressure', '_velocity']
    
    def __init__(self):
        self._velocity = None
        self._pressure = None

    @property
    def velocity(self) -> list: return self._get_safe("velocity")
    @velocity.setter
    def velocity(self, v: list):
        if v is not None and len(v) != 3: raise ValueError("velocity must have 3 items")
        self._set_safe("velocity", v, list)
    @property
    def pressure(self) -> float: return self._get_safe("pressure")
    @pressure.setter
    def pressure(self, v: float): self._set_safe("pressure", v, float)

@dataclass
class SimParamsInput(ValidatedContainer):
    __slots__ = ['_output_interval', '_time_step', '_total_time']
    
    def __init__(self):
        self._time_step = None
        self._total_time = None
        self._output_interval = None

    @property
    def time_step(self) -> float: return self._get_safe("time_step")
    @time_step.setter
    def time_step(self, v: float):
        if v <= 0: raise ValueError("time_step must be > 0")
        self._set_safe("time_step", v, float)
    @property
    def total_time(self) -> float: return self._get_safe("total_time")
    @total_time.setter
    def total_time(self, v: float):
        if v <= 0: raise ValueError("total_time must be > 0")
        self._set_safe("total_time", v, float)
    @property
    def output_interval(self) -> int: return self._get_safe("output_interval")
    @output_interval.setter
    def output_interval(self, v: int):
        if v < 1: raise ValueError("output_interval must be >= 1")
        self._set_safe("output_interval", v, int)

@dataclass
class BoundaryConditionItem(ValidatedContainer):
    __slots__ = ['_location', '_type', '_values']
    
    def __init__(self, location: str, type: str, values: dict):
        self.location = location
        self.type = type
        self.values = values

    @property
    def location(self) -> str: return self._get_safe("location")
    @location.setter
    def location(self, v: str):
        valid = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "wall"]
        if v not in valid: raise ValueError(f"Invalid location: {v}")
        self._set_safe("location", v, str)

    @property
    def type(self) -> str: return self._get_safe("type")
    @type.setter
    def type(self, v: str):
        valid = ["no-slip", "free-slip", "inflow", "outflow", "pressure"]
        if v not in valid: raise ValueError(f"Invalid type: {v}")
        self._set_safe("type", v, str)

    @property
    def values(self) -> dict: return self._get_safe("values")
    @values.setter
    def values(self, v: dict): self._set_safe("values", v, dict)

@dataclass
class BoundaryConditionsInput(ValidatedContainer):
    __slots__ = ['_items']
    
    def __init__(self): self._items = None

    @property
    def items(self) -> list[BoundaryConditionItem]: return self._get_safe("items")
    @items.setter
    def items(self, v: list):
        processed = [bc if isinstance(bc, BoundaryConditionItem) else BoundaryConditionItem(**bc) for bc in v]
        self._set_safe("items", processed, list)

@dataclass
class MaskInput(ValidatedContainer):
    __slots__ = ['_data']
    
    def __init__(self): self._data = None

    @property
    def data(self) -> list: return self._get_safe("data")
    @data.setter
    def data(self, v: list):
        if not all(val in {-1, 0, 1} for val in v):
            raise ValueError("Mask contains invalid values. Only -1, 0, 1 allowed.")
        self._set_safe("data", v, list)

@dataclass
class ExternalForcesInput(ValidatedContainer):
    __slots__ = ['_force_vector']
    
    def __init__(self): self._force_vector = None

    @property
    def force_vector(self) -> list: return self._get_safe("force_vector")
    @force_vector.setter
    def force_vector(self, v: list):
        if len(v) != 3: raise ValueError("force_vector must have 3 items")
        self._set_safe("force_vector", v, list)

# =========================================================
# 2. THE UNIVERSAL INPUT CONTAINER
# =========================================================

@dataclass
class SolverInput(ValidatedContainer):
    __slots__ = [
        'boundary_conditions',
        'domain_configuration',
        'external_forces',
        'fluid_properties',
        'grid',
        'initial_conditions',
        'mask',
        'physical_constraints',
        'simulation_parameters'
    ]
    
    def __init__(self):
        for slot in self.__slots__: object.__setattr__(self, slot, None)

    @classmethod
    def from_dict(cls, data: dict) -> "SolverInput":
        obj = cls()
        # Initialize sub-containers
        obj.domain_configuration = DomainConfigInput()
        obj.grid = GridInput()
        obj.fluid_properties = FluidInput()
        obj.initial_conditions = InitialConditionsInput()
        obj.simulation_parameters = SimParamsInput()
        obj.external_forces = ExternalForcesInput()
        obj.mask = MaskInput()
        obj.boundary_conditions = BoundaryConditionsInput()
        obj.physical_constraints = PhysicalConstraintsInput()

        # Ingestion logic
        obj.boundary_conditions.items = data["boundary_conditions"]
        
        dc = data["domain_configuration"]
        obj.domain_configuration.type = dc["type"]
        if "reference_velocity" in dc and dc["reference_velocity"] is not None:
            obj.domain_configuration.reference_velocity = dc["reference_velocity"]
        
        g = data["grid"]
        for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]:
            setattr(obj.grid, k, g[k])
            
        f = data["fluid_properties"]
        obj.fluid_properties.density = f["density"]
        obj.fluid_properties.viscosity = f["viscosity"]
        
        ic = data["initial_conditions"]
        obj.initial_conditions.velocity = ic["velocity"]
        obj.initial_conditions.pressure = ic["pressure"]
        
        sp = data["simulation_parameters"]
        obj.simulation_parameters.time_step = sp["time_step"]
        obj.simulation_parameters.total_time = sp["total_time"]
        obj.simulation_parameters.output_interval = sp["output_interval"]
        
        obj.external_forces.force_vector = data["external_forces"]["force_vector"]
        obj.mask.data = data["mask"]
        obj.boundary_conditions.items = data["boundary_conditions"]
        
        pc = data["physical_constraints"]
        obj.physical_constraints.min_velocity = pc["min_velocity"]
        obj.physical_constraints.max_velocity = pc["max_velocity"]
        obj.physical_constraints.min_pressure = pc["min_pressure"]
        obj.physical_constraints.max_pressure = pc["max_pressure"]
        
        return obj

    def to_dict(self) -> dict:
        domain_cfg = {"type": self.domain_configuration.type}
        if hasattr(self.domain_configuration, '_reference_velocity') and self.domain_configuration._reference_velocity is not None:
            domain_cfg["reference_velocity"] = self.domain_configuration.reference_velocity
            
        return {
            "physical_constraints": {
                "min_velocity": self.physical_constraints.min_velocity,
                "max_velocity": self.physical_constraints.max_velocity,
                "min_pressure": self.physical_constraints.min_pressure,
                "max_pressure": self.physical_constraints.max_pressure
            },
            "domain_configuration": domain_cfg,
            "grid": {k: getattr(self.grid, k) for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "nx", "ny", "nz"]},
            "fluid_properties": {"density": self.fluid_properties.density, "viscosity": self.fluid_properties.viscosity},
            "initial_conditions": {"velocity": self.initial_conditions.velocity, "pressure": self.initial_conditions.pressure},
            "simulation_parameters": {
                "time_step": self.simulation_parameters.time_step, 
                "total_time": self.simulation_parameters.total_time, 
                "output_interval": self.simulation_parameters.output_interval
            },
            "boundary_conditions": [{"location": bc.location, "type": bc.type, "values": bc.values} for bc in self.boundary_conditions.items],
            "mask": self.mask.data,
            "external_forces": {"force_vector": self.external_forces.force_vector}
        }
