"""
src/generate_previews.py
Preview Generation Module.
Extracts 2D cross-sectional slices from 3D state arrays and generates visual preview artifacts.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("Solver.Previews")

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not found. Preview generation falling back to raw NumPy slice export.")


def generate_snapshot_preview(state, output_dir: str, step: int) -> str:
    """
    Extracts a mid-plane 2D cross-section of pressure and velocity magnitude,
    rendering a PNG preview image into the target output directory.

    Args:
        state: SolverState instance
        output_dir: Target directory path for saving preview files
        step: Current simulation step iteration index

    Returns:
        Relative path string to the generated preview file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    mid_z = state.nz // 2
    
    # Extract 2D mid-plane slices
    u_slice = state.fields[0, :, :, mid_z]
    v_slice = state.fields[1, :, :, mid_z]
    p_slice = state.fields[3, :, :, mid_z]
    vel_mag_slice = np.sqrt(u_slice**2 + v_slice**2)

    file_name = f"preview_step_{step:06d}.png"
    file_path = out_path / file_name

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Velocity magnitude contour
        im0 = axes[0].imshow(
            vel_mag_slice.T, origin="lower", cmap="viridis",
            extent=[state.x_min, state.x_max, state.y_min, state.y_max]
        )
        axes[0].set_title(f"Velocity Mag (Z-slice, Step {step})")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        fig.colorbar(im0, ax=axes[0])

        # Pressure field contour
        im1 = axes[1].imshow(
            p_slice.T, origin="lower", cmap="coolwarm",
            extent=[state.x_min, state.x_max, state.y_min, state.y_max]
        )
        axes[1].set_title(f"Pressure Field (Z-slice, Step {step})")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        fig.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(file_path, dpi=100)
        plt.close(fig)
    else:
        # Fallback: Save raw array values to CSV if matplotlib is unavailable
        raw_file_name = f"preview_step_{step:06d}.csv"
        file_path = out_path / raw_file_name
        np.savetxt(file_path, vel_mag_slice, delimiter=",")

    logger.debug(f"Generated simulation preview slice: {file_path}")
    return str(file_name)
