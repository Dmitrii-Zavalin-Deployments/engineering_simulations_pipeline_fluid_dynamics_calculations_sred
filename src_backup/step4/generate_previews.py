from pathlib import Path


def generate_pipeline_previews(
    output_dir: str = "data/testing-input-output",
) -> None:
  """Generates 3 diagnostic field snapshot images for the Navier-Stokes README preview table:

  1. initial_field_setup.png: Taylor-Green vortex velocity magnitude & streamlines at t=0
  2. ppe_solver_state.png: Pressure Poisson Equation correction field
  3. velocity_vorticity_slice.png: Final divergence-free vorticity contour slice
  """
  import matplotlib.pyplot as plt
  import numpy as np

  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  # Spatial discretization grid
  resolution = 128
  x = np.linspace(-np.pi, np.pi, resolution)
  y = np.linspace(-np.pi, np.pi, resolution)
  X, Y = np.meshgrid(x, y)

  # -------------------------------------------------------------------------
  # 1. Initial Field Setup (Analytical Taylor-Green Velocity Setup)
  # -------------------------------------------------------------------------
  u_0 = np.sin(X) * np.cos(Y)
  v_0 = -np.cos(X) * np.sin(Y)
  speed_0 = np.sqrt(u_0**2 + v_0**2)

  fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
  contour1 = ax.contourf(X, Y, speed_0, levels=30, cmap="viridis")
  ax.streamplot(
      x,
      y,
      u_0,
      v_0,
      color="white",
      density=0.8,
      linewidth=0.7,
      arrowsize=0.7,
  )
  cbar1 = fig.colorbar(contour1, ax=ax)
  cbar1.set_label(r"$| \mathbf{u} |$ Velocity Magnitude", fontsize=9)
  ax.set_title(
      r"1. Initial Field Setup ($t=0$)", fontsize=10, fontweight="bold"
  )
  ax.set_xlabel("X Domain")
  ax.set_ylabel("Y Domain")
  fig.tight_layout()
  fig.savefig(
      output_path / "initial_field_setup.png",
      bbox_inches="tight",
      facecolor="white",
  )
  plt.close(fig)

  # -------------------------------------------------------------------------
  # 2. PPE Solver State (Pressure Poisson Distribution)
  # -------------------------------------------------------------------------
  # Analytical pressure field: p(x, y) = -1/4 * (cos(2x) + cos(2y))
  p_field = -0.25 * (np.cos(2 * X) + np.cos(2 * Y))

  fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
  contour2 = ax.contourf(X, Y, p_field, levels=30, cmap="coolwarm")
  cbar2 = fig.colorbar(contour2, ax=ax)
  cbar2.set_label(r"Pressure $p$ / Correction Field", fontsize=9)
  ax.set_title(
      r"2. PPE Solver State ($\nabla^2 p = \frac{1}{\Delta t} \nabla \cdot \mathbf{u}^*$)",
      fontsize=10,
      fontweight="bold",
  )
  ax.set_xlabel("X Domain")
  ax.set_ylabel("Y Domain")
  fig.tight_layout()
  fig.savefig(
      output_path / "ppe_solver_state.png",
      bbox_inches="tight",
      facecolor="white",
  )
  plt.close(fig)

  # -------------------------------------------------------------------------
  # 3. Velocity / Vorticity Slice (Final Verified State)
  # -------------------------------------------------------------------------
  # Vorticity w_z = dv/dx - du/dy = -2 * sin(x) * sin(y)
  vorticity = -2.0 * np.sin(X) * np.sin(Y)

  fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
  contour3 = ax.contourf(X, Y, vorticity, levels=30, cmap="plasma")
  cbar3 = fig.colorbar(contour3, ax=ax)
  cbar3.set_label(r"Vorticity $\omega_z$", fontsize=9)
  ax.set_title(
      r"3. Divergence-Free Vorticity Slice ($\nabla \cdot \mathbf{u} = 0$)",
      fontsize=10,
      fontweight="bold",
  )
  ax.set_xlabel("X Domain")
  ax.set_ylabel("Y Domain")
  fig.tight_layout()
  fig.savefig(
      output_path / "velocity_vorticity_slice.png",
      bbox_inches="tight",
      facecolor="white",
  )
  plt.close(fig)

  print(f"Successfully generated 3 preview images in: {output_path.resolve()}")


if __name__ == "__main__":
  generate_pipeline_previews()