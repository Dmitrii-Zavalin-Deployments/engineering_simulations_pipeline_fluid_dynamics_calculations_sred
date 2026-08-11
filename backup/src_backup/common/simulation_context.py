from dataclasses import dataclass

from src.common.solver_config import SolverConfig
from src.common.solver_input import SolverInput


@dataclass
class SimulationContext:
    """
    Acts as the primary dependency injection container for the solver.
    """
    input_data: SolverInput
    config: SolverConfig

    @classmethod
    def create(cls, input_dict: dict, config_dict: dict) -> "SimulationContext":
        # Rule 4: Extract physical data first
        input_data = SolverInput.from_dict(input_dict)

        # Rule 5: Ensure 'dt' doesn't contaminate the static config object
        config_dict.pop("dt", None)
        config = SolverConfig(**config_dict)

        return cls(input_data=input_data, config=config)
