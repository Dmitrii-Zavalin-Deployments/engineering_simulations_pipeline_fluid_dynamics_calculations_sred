# tests/helpers/solver_step4_output_dummy.py

from src.common.field_schema import FI
from src.common.simulation_context import SimulationContext
from src.common.solver_config import SolverConfig
from src.step1.orchestrate_step1 import orchestrate_step1
from src.step2.orchestrate_step2 import orchestrate_step2
from tests.helpers.solver_input_schema_dummy import create_validated_input


def make_step4_output_dummy(nx: int = 4, ny: int = 4, nz: int = 4):
    """
    Returns the Global SolverState after Step 3 (Converged) and 
    Elasticity (Committed). This is the 'Archival' state.
    
    Compliance:
    - Rule 9: Foundation buffers match Trial buffers (Post-Commit).
    - Rule 4: Includes manifest metadata for orchestrate_step4 (Archivist).
    """
    # 1. Setup Context
    MOCK_CONFIG = {
        "ppe_tolerance": 1e-6,
        "ppe_max_iter": 1000,
        "dt_min_limit": 1e-6,
        "ppe_max_retries": 5
    }
    
    input_dummy = create_validated_input(nx=nx, ny=ny, nz=nz)
    config_obj = SolverConfig(**MOCK_CONFIG)
    context = SimulationContext(input_data=input_dummy, config=config_obj)
    
    # 2. Build Base State (Allocates Fields)
    state = orchestrate_step2(orchestrate_step1(context))
    data = state.fields.data
    
    # 3. MOCK THE COMMIT (The result of elasticity.stabilization)
    # Target converged values
    target_vel = 0.50
    target_p = 0.012

    # A. Trial Buffers (Where the math happened)
    data[:, FI.VX_STAR] = target_vel
    data[:, FI.VY_STAR] = target_vel
    data[:, FI.VZ_STAR] = target_vel
    data[:, FI.P_NEXT]  = target_p

    # B. Foundation Buffers (Where the data was committed)
    # This reflects Rule 9: The Trial data is now the Official data.
    data[:, FI.VX] = target_vel
    data[:, FI.VY] = target_vel
    data[:, FI.VZ] = target_vel
    data[:, FI.P]  = target_p

    # 4. Set Runtime Metadata
    state.iteration = input_dummy.simulation_parameters.output_interval # Ensure it triggers save
    state.time = state.iteration * input_dummy.simulation_parameters.time_step
    
    # 5. Manifest Integrity (For Step 4 Archivist)
    if not hasattr(state, 'manifest'):
        class ManifestDummy: pass
        state.manifest = ManifestDummy()
    
    state.manifest.saved_snapshots = []
    state.manifest.output_directory = "output/"
    state.ready_for_time_loop = True
    
    return state