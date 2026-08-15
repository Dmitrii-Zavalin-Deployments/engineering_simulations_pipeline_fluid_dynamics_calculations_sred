python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

print('==========================================================================')
print('      FORENSIC AUDIT & DIAGNOSTIC SUITE FOR NAVIER-STOKES PIPELINE')
print('==========================================================================')

print('\n[1] INSPECTING SOLVER STATE INITIALIZATION & FIELD ATTRIBUTES')
print('--------------------------------------------------------------------------')
try:
    from src.state import SolverState
    print('SolverState imported successfully.')
    state_attrs = dir(SolverState)
    print('SolverState relevant attributes:', [a for a in state_attrs if not a.startswith('__')])
except Exception as e:
    print(f'Error inspecting SolverState: {e}')

print('\n[2] SMOKING-GUN SOURCE AUDIT: src/main.py (Time loop & Step execution)')
print('--------------------------------------------------------------------------')
main_path = Path('src/main.py')
if main_path.is_file():
    lines = main_path.read_text(encoding='utf-8').splitlines()
    for idx, line in enumerate(lines, 1):
        print(f'{idx:4d} | {line}')

print('\n[3] SMOKING-GUN SOURCE AUDIT: src/cpp_gate.py (Step & Sync execution)')
print('--------------------------------------------------------------------------')
cpp_gate_path = Path('src/cpp_gate.py')
if cpp_gate_path.is_file():
    lines = cpp_gate_path.read_text(encoding='utf-8').splitlines()
    for idx, line in enumerate(lines, 1):
        print(f'{idx:4d} | {line}')

print('\n[4] INSPECTING PYBIND11 C++ MODULE BINDINGS AND SYNC EXPOSURE')
print('--------------------------------------------------------------------------')
try:
    import navier_stokes_cpp
    print('navier_stokes_cpp C++ module imported successfully.')
    print('Module contents:', dir(navier_stokes_cpp))
    if hasattr(navier_stokes_cpp, 'NavierStokesSolver'):
        solver_methods = dir(navier_stokes_cpp.NavierStokesSolver)
        print('NavierStokesSolver exposed methods:', [m for m in solver_methods if not m.startswith('__')])
except Exception as e:
    print(f'Error inspecting C++ module: {e}')

print('\n==========================================================================')
print('                      FORENSIC AUDIT COMPLETE')
print('==========================================================================')
"