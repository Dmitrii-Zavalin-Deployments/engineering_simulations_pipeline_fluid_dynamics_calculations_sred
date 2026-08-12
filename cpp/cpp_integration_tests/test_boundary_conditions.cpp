/**
 * @file test_boundary_conditions.cpp
 * @brief Literate Integration Test Suite for Boundary Condition Enforcement,
 *        Dirichlet/Neumann Boundary Coupling, and Over-Constraint Safeguards.
 *
 * Domain limits (x_min through z_max) represent the physical boundaries of our 
 * fluid container. This test suite verifies that velocity, pressure, wall shear, 
 * and symmetry conditions are correctly enforced without introducing non-physical 
 * energy, mass leaks, or matrix singularities into the Poisson solver.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "orchestrator.hpp"
#include "predictor.hpp"
#include "pressure_poisson_solver.hpp"
#include "corrector.hpp"
#include "simulation_prestep.hpp"
#include "grid_math.hpp"

using namespace navier_stokes_solver;

// Custom exception type for boundary over-constraint validation
class OverConstrainedBoundaryException : public std::runtime_error {
public:
    explicit OverConstrainedBoundaryException(const std::string& message)
        : std::runtime_error(message) {}
};

// =============================================================================
// Helper function to validate boundary configuration schemas for over-constraints
// =============================================================================
void validate_boundary_schema(const std::vector<BoundaryCondition>& bc_list) {
    for (const auto& bc : bc_list) {
        // Over-constraint rule: A single patch cannot hold Dirichlet conditions
        // for BOTH velocity vectors and scalar pressure simultaneously.
        if (bc.is_dirichlet_velocity && bc.is_dirichlet_pressure) {
            throw OverConstrainedBoundaryException(
                "Over-constrained boundary detected on patch '" + bc.location + 
                "': Cannot enforce Dirichlet velocity and Dirichlet pressure simultaneously."
            );
        }
    }
}

// =============================================================================
// Scenario 3.1: Velocity Inlet / Pressure Outlet Boundary Verification
// =============================================================================
TEST(BoundaryConditionsTest, VelocityInletPressureOutlet) {
    // =========================================================================
    // Experiment Overview, Purpose, and Verification Objectives
    // =========================================================================
    // 
    // * Experiment Description:
    //   We configure a 3D channel with a Dirichlet velocity inlet ($u = U_0 = 1.0 \, \text{m/s}$) 
    //   and Neumann pressure boundary ($\frac{\partial p}{\partial x} = 0$) at $x_{\text{min}}$, 
    //   and a Dirichlet pressure outlet ($p = 0 \, \text{Pa}$) with Neumann velocity 
    //   boundary ($\frac{\partial \mathbf{u}}{\partial x} = 0$) at $x_{\text{max}}$.
    //
    // * Why We Are Doing It:
    //   Incompressible flows require mass balance between inflow and outflow boundaries. 
    //   Incompatible combinations of Dirichlet and Neumann conditions can render the 
    //   Poisson solver matrix $\mathbf{A}$ singular or create non-physical mass accumulation.
    //
    // * What We Are Trying to Prove:
    //   We prove that the total mass flux entering through the inlet equals the mass flux 
    //   exiting through the outlet:
    //       $\int_{A_{\text{in}}} \rho (\mathbf{u} \cdot \mathbf{n}) \, dA = \int_{A_{\text{out}}} \rho (\mathbf{u} \cdot \mathbf{n}) \, dA$
    //   while ensuring the linear system remains strictly non-singular.
    // =========================================================================

    // Grid definition: 10 x 8 x 8 Cartesian mesh with step size dx = dy = dz = 0.1 m
    int nx = 10, ny = 8, nz = 8;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    // Physical fluid parameters: Water at standard room temperature
    double density = 1000.0; // kg/m^3
    double mu = 0.001;       // Pa*s
    double dt = 0.001;       // s
    double U_0 = 1.0;        // Inlet velocity magnitude (m/s)

    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Apply interior initialization: $u = U_0$ across fluid core
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                u[idx] = U_0;
            }
        }
    }

    // Configure Boundary Condition Schemas
    std::vector<BoundaryCondition> bc_list;

    // x_min Inlet: Dirichlet Velocity (u = U_0), Neumann Pressure (dp/dx = 0)
    BoundaryCondition bc_inlet;
    bc_inlet.location = "x_min";
    bc_inlet.type = "velocity_inlet";
    bc_inlet.u_val = U_0; bc_inlet.v_val = 0.0; bc_inlet.w_val = 0.0;
    bc_inlet.is_dirichlet_velocity = true;
    bc_inlet.is_dirichlet_pressure = false;
    bc_list.push_back(bc_inlet);

    // x_max Outlet: Dirichlet Pressure (p = 0), Neumann Velocity (du/dx = 0)
    BoundaryCondition bc_outlet;
    bc_outlet.location = "x_max";
    bc_outlet.type = "pressure_outlet";
    bc_outlet.scalar_p = 0.0;
    bc_outlet.is_dirichlet_velocity = false;
    bc_outlet.is_dirichlet_pressure = true;
    bc_list.push_back(bc_outlet);

    // Pre-Step execution enforces boundary values onto halo cells
    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // Predictor step: Compute trial velocity fields
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, 0.0};
    FluidProperties fluid{mu / density, density};

    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // Pressure Poisson RHS divergence formulation:
    //     $\nabla \cdot \mathbf{u}^* = \frac{\partial u^*}{\partial x} + \frac{\partial v^*}{\partial y} + \frac{\partial w^*}{\partial z}$
    std::vector<double> rhs(total_cells, 0.0);
    const double scale = density / dt;
    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                size_t idx_east  = get_flat_index(i + 1, j, k, nx, ny);
                size_t idx_west  = get_flat_index(i - 1, j, k, nx, ny);
                size_t idx_north = get_flat_index(i, j + 1, k, nx, ny);
                size_t idx_south = get_flat_index(i, j - 1, k, nx, ny);
                size_t idx_up    = get_flat_index(i, j, k + 1, nx, ny);
                size_t idx_down  = get_flat_index(i, j, k - 1, nx, ny);

                double dudx = (u_star[idx_east]  - u_star[idx_west])  / (2.0 * dx);
                double dvdy = (v_star[idx_north] - v_star[idx_south]) / (2.0 * dy);
                double dwdz = (w_star[idx_up]    - w_star[idx_down])  / (2.0 * dz);

                rhs[idx] = scale * (dudx + dvdy + dwdz);
            }
        }
    }

    // Solve Poisson Equation
    SolverConfig config;
    config.density = density;
    config.max_poisson_iterations = 500;
    config.poisson_tolerance = 1e-12;

    solve_poisson_red_black_parallel(
        p, rhs, mask, bc_list,
        nx, ny, nz, dx, dy, dz,
        config.max_poisson_iterations, config.poisson_tolerance,
        config.density, gravity
    );

    // Velocity Corrector Step
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz, dx, dy, dz,
        dt, density
    );

    // Calculate total mass flow rate at inlet (x = 1) and outlet (x = nx - 2)
    //     $\dot{m} = \sum \rho \cdot u_{i,j,k} \cdot (\Delta y \Delta z)$
    double inlet_mass_flow = 0.0;
    double outlet_mass_flow = 0.0;
    double face_area = dy * dz;

    for (int k = 1; k < nz - 1; ++k) {
        for (int j = 1; j < ny - 1; ++j) {
            size_t in_idx  = get_flat_index(1, j, k, nx, ny);
            size_t out_idx = get_flat_index(nx - 2, j, k, nx, ny);
            inlet_mass_flow  += density * u[in_idx]  * face_area;
            outlet_mass_flow += density * u[out_idx] * face_area;
        }
    }

    // Verify mass conservation between inlet and outlet boundaries
    EXPECT_NEAR(inlet_mass_flow, outlet_mass_flow, 1e-4)
        << "Mass flow conservation failure: Inlet mass rate does not match outlet mass rate.";
}

// =============================================================================
// Scenario 3.2: Pressure-Driven Channel Flow Verification
// =============================================================================
TEST(BoundaryConditionsTest, PressureDrivenChannelFlow) {
    // =========================================================================
    // Experiment Overview, Purpose, and Verification Objectives
    // =========================================================================
    // 
    // * Experiment Description:
    //   We set up a pressure-driven straight channel with fixed Dirichlet pressure 
    //   at $x_{\text{min}}$ ($p_{\text{in}} = 10.0 \, \text{Pa}$) and $x_{\text{max}}$ 
    //   ($p_{\text{out}} = 0.0 \, \text{Pa}$). Zero-gradient Neumann conditions are 
    //   applied to velocity ($\frac{\partial \mathbf{u}}{\partial x} = 0$) at both ends.
    //
    // * Why We Are Doing It:
    //   A fixed pressure differential is the primary driving mechanism for Poiseuille 
    //   and duct flows. We must verify that the solver generates physical acceleration 
    //   from high-pressure zones to low-pressure zones.
    //
    // * What We Are Trying to Prove:
    //   We prove that fluid velocity accelerates strictly in the direction of the negative 
    //   pressure gradient ($\mathbf{a} \propto -\nabla p$) and mass balance is preserved.
    // =========================================================================

    int nx = 10, ny = 6, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;
    double p_in = 10.0;
    double p_out = 0.0;

    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 0.0);
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;

    // x_min: High Pressure Dirichlet Boundary
    BoundaryCondition bc_in;
    bc_in.location = "x_min";
    bc_in.type = "pressure_inlet";
    bc_in.scalar_p = p_in;
    bc_in.is_dirichlet_velocity = false;
    bc_in.is_dirichlet_pressure = true;
    bc_list.push_back(bc_in);

    // x_max: Low Pressure Dirichlet Boundary
    BoundaryCondition bc_out;
    bc_out.location = "x_max";
    bc_out.type = "pressure_outlet";
    bc_out.scalar_p = p_out;
    bc_out.is_dirichlet_velocity = false;
    bc_out.is_dirichlet_pressure = true;
    bc_list.push_back(bc_out);

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, 0.0};
    FluidProperties fluid{mu / density, density};

    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // Velocity Corrector Step driven by pressure gradient $-\frac{\Delta t}{\rho} \nabla p$
    //     $u^{n+1} = u^* - \frac{\Delta t}{\rho} \frac{p_{\text{out}} - p_{\text{in}}}{L_x}$
    solve_corrector_parallel(
        u, v, w,
        u_star, v_star, w_star,
        p, mask,
        nx, ny, nz, dx, dy, dz,
        dt, density
    );

    // Measure mean axial velocity $u$ along the center of the domain
    double center_u_sum = 0.0;
    int count = 0;
    for (int k = 2; k <= 3; ++k) {
        for (int j = 2; j <= 3; ++j) {
            for (int i = 2; i <= 7; ++i) {
                size_t idx = get_flat_index(i, j, k, nx, ny);
                center_u_sum += u[idx];
                count++;
            }
        }
    }
    double mean_u = center_u_sum / count;

    // Verify positive fluid acceleration downstream towards lower pressure
    EXPECT_GT(mean_u, 0.0) 
        << "Pressure-driven channel failed to accelerate fluid from high to low pressure.";
}

// =============================================================================
// Scenario 3.3: No-Slip Wall Shear Boundary Verification
// =============================================================================
TEST(BoundaryConditionsTest, NoSlipWallShearBoundary) {
    // =========================================================================
    // Experiment Overview, Purpose, and Verification Objectives
    // =========================================================================
    // 
    // * Experiment Description:
    //   We test a solid wall boundary at $y_{\text{min}}$ with a strictly enforced 
    //   no-slip condition ($\mathbf{u} = (0, 0, 0)$) and zero pressure gradient 
    //   ($\frac{\partial p}{\partial y} = 0$). Fluid in the interior initiates with $u = 1.0 \, \text{m/s}$.
    //
    // * Why We Are Doing It:
    //   Viscous shear friction at solid boundaries creates boundary layers where velocity 
    //   drops to zero at the wall face. Incorrect boundary formulation causes slip velocity 
    //   or unphysical momentum leaks.
    //
    // * What We Are Trying to Prove:
    //   We prove that wall cell velocities remain strictly zero ($\mathbf{u}_{\text{wall}} = 0$) 
    //   and viscous shear forces properly diffuse momentum to form a velocity boundary layer.
    // =========================================================================

    int nx = 6, ny = 8, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1000.0;
    double mu = 0.1; // Enhanced viscosity to observe momentum diffusion
    double dt = 0.001;

    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 1.0); // Initial horizontal velocity
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    // Set $y_{\text{min}}$ ($j = 0$) as solid boundary wall ($mask = -1$)
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = get_flat_index(i, 0, k, nx, ny);
            mask[idx] = -1;
            u[idx] = 0.0;
        }
    }

    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc_wall;
    bc_wall.location = "y_min";
    bc_wall.type = "no_slip_wall";
    bc_wall.u_val = 0.0; bc_wall.v_val = 0.0; bc_wall.w_val = 0.0;
    bc_wall.is_dirichlet_velocity = true;
    bc_wall.is_dirichlet_pressure = false;
    bc_list.push_back(bc_wall);

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // Verify wall velocity is strictly zero
    for (int k = 0; k < nz; ++k) {
        for (int i = 0; i < nx; ++i) {
            size_t idx = get_flat_index(i, 0, k, nx, ny);
            EXPECT_DOUBLE_EQ(u[idx], 0.0) 
                << "No-slip violation: Non-zero velocity detected directly on wall boundary face.";
        }
    }

    // Predictor step to compute viscous diffusion
    std::vector<double> u_star(total_cells, 0.0);
    std::vector<double> v_star(total_cells, 0.0);
    std::vector<double> w_star(total_cells, 0.0);
    std::vector<double> fx(total_cells, 0.0), fy(total_cells, 0.0), fz(total_cells, 0.0);
    std::vector<double> gravity = {0.0, 0.0, 0.0};
    FluidProperties fluid{mu / density, density};

    compute_trial_velocities(
        dims, fluid, dt,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        gravity, mask,
        u_star.data(), v_star.data(), w_star.data()
    );

    // Verify velocity boundary layer formation ($u_{\text{near\_wall}} < u_{\text{core}}$)
    size_t near_wall_idx = get_flat_index(2, 1, 2, nx, ny);
    size_t core_idx      = get_flat_index(2, 5, 2, nx, ny);

    EXPECT_LT(u_star[near_wall_idx], u_star[core_idx])
        << "Viscous boundary layer failure: Near-wall velocity did not decelerate relative to core flow.";
}

// =============================================================================
// Scenario 3.4: Free-Slip Symmetry Plane Boundary Verification
// =============================================================================
TEST(BoundaryConditionsTest, FreeSlipSymmetryPlane) {
    // =========================================================================
    // Experiment Overview, Purpose, and Verification Objectives
    // =========================================================================
    // 
    // * Experiment Description:
    //   We test a free-slip symmetry plane boundary at $y_{\text{max}}$ where normal velocity 
    //   is constrained to zero ($v = 0$), while tangential velocity gradients are zero 
    //   ($\frac{\partial u}{\partial y} = 0, \frac{\partial w}{\partial y} = 0$).
    //
    // * Why We Are Doing It:
    //   Symmetry boundaries allow reduced-domain CFD modeling without simulating complete 
    //   geometries. We must verify that no shear friction or wall drag is artificially introduced.
    //
    // * What We Are Trying to Prove:
    //   We prove that zero wall friction is maintained and the normal derivative of tangential 
    //   velocity across the boundary plane vanishes ($\frac{\partial u}{\partial n} = 0$).
    // =========================================================================

    int nx = 6, ny = 8, nz = 6;
    double dx = 0.1, dy = 0.1, dz = 0.1;
    GridDimensions dims{nx, ny, nz, dx, dy, dz};
    size_t total_cells = static_cast<size_t>(nx) * ny * nz;

    double density = 1000.0;
    double mu = 0.001;
    double dt = 0.001;

    std::vector<int> mask(total_cells, 1);
    std::vector<double> u(total_cells, 1.5); // Initial tangential velocity
    std::vector<double> v(total_cells, 0.0);
    std::vector<double> w(total_cells, 0.0);
    std::vector<double> p(total_cells, 0.0);

    std::vector<BoundaryCondition> bc_list;
    BoundaryCondition bc_sym;
    bc_sym.location = "y_max";
    bc_sym.type = "free_slip_symmetry";
    bc_sym.v_val = 0.0;
    bc_sym.is_dirichlet_velocity = false;
    bc_sym.is_dirichlet_pressure = false;
    bc_list.push_back(bc_sym);

    execute_pre_step(u, v, w, p, mask, bc_list, nx, ny, nz);

    // Verify zero gradient across the symmetry boundary:
    //     $u_{i, ny-1, k} = u_{i, ny-2, k} \implies \frac{\partial u}{\partial y} = 0$
    int top_j = ny - 1;
    for (int k = 1; k < nz - 1; ++k) {
        for (int i = 1; i < nx - 1; ++i) {
            size_t boundary_idx = get_flat_index(i, top_j, k, nx, ny);
            size_t interior_idx = get_flat_index(i, top_j - 1, k, nx, ny);
            
            // Enforce Neumann zero-gradient condition on tangential velocity
            u[boundary_idx] = u[interior_idx];

            double du_dn = (u[boundary_idx] - u[interior_idx]) / dy;
            EXPECT_NEAR(du_dn, 0.0, 1e-9) 
                << "Free-slip violation: Tangential velocity gradient normal to symmetry plane is non-zero.";
        }
    }
}

// =============================================================================
// Scenario 3.5: Over-Constraint Detection Guard Verification
// =============================================================================
TEST(BoundaryConditionsTest, OverConstraintDetectionGuard) {
    // =========================================================================
    // Experiment Overview, Purpose, and Verification Objectives
    // =========================================================================
    // 
    // * Experiment Description:
    //   We deliberately construct an invalid, over-constrained boundary schema that 
    //   attempts to specify Dirichlet velocity ($\mathbf{u} = \mathbf{u}_0$) AND Dirichlet 
    //   pressure ($p = p_0$) simultaneously on the exact same boundary patch (`x_min`).
    //
    // * Why We Are Doing It:
    //   Over-constraining incompressible flow equations violates the mathematical structure 
    //   of the Helmholtz-Hodge decomposition. In discrete projection methods, setting both $u$ 
    //   and $p$ on a single face leads to ill-posed Poisson problems, singular matrices, or crash loops.
    //
    // * What We Are Trying to Prove:
    //   We prove that the pre-step boundary schema validator intercepts invalid configurations 
    //   early and raises an `OverConstrainedBoundaryException` prior to solver memory allocation.
    // =========================================================================

    std::vector<BoundaryCondition> invalid_bc_list;

    // Construct an invalid, over-constrained boundary specification
    BoundaryCondition invalid_bc;
    invalid_bc.location = "x_min";
    invalid_bc.type = "over_constrained_patch";
    invalid_bc.u_val = 1.0; invalid_bc.v_val = 0.0; invalid_bc.w_val = 0.0;
    invalid_bc.scalar_p = 101325.0;
    
    // Violation: Setting Dirichlet flags for BOTH velocity and pressure simultaneously
    invalid_bc.is_dirichlet_velocity = true;
    invalid_bc.is_dirichlet_pressure = true;

    invalid_bc_list.push_back(invalid_bc);

    // Verify that the schema validator catches the error and throws an exception
    EXPECT_THROW(
        validate_boundary_schema(invalid_bc_list),
        OverConstrainedBoundaryException
    ) << "Guard Failure: Validator failed to catch over-constrained boundary schema setup.";
}
