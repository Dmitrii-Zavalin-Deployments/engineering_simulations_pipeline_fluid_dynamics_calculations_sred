#include "orchestrator.hpp"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace ops {

NavierStokesOrchestrator::NavierStokesOrchestrator(const GridDimensions& dims, const SolverConfig& config)
    : dims_(dims), config_(config), total_cells_(dims.nx * dims.ny * dims.nz) {
    u_star_.resize(total_cells_, 0.0);
    v_star_.resize(total_cells_, 0.0);
    w_star_.resize(total_cells_, 0.0);
    rhs_.resize(total_cells_, 0.0);
}

void NavierStokesOrchestrator::step(
    double dt,
    double mu,
    const std::vector<double>& fx,
    const std::vector<double>& fy,
    const std::vector<double>& fz,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p
) {
    // 1. PRE-STEP: Apply static Dirichlet velocity/pressure conditions on walls (mask == -1) and solids (mask == 0)
    apply_pre_step_bcs(mask, bc_list, u, v, w, p);

    // 2. PREDICTOR STEP: Compute trial velocities (u*, v*, w*) for active fluid cells (mask == 1)
    FluidProperties fluid{dt, mu / config_.density};
    compute_trial_velocities(
        dims_, fluid,
        u.data(), v.data(), w.data(),
        fx.data(), fy.data(), fz.data(),
        mask,
        u_star_.data(), v_star_.data(), w_star_.data()
    );

    // 3. PRESSURE POISSON STEP: Compute RHS divergence and solve pressure field iteratively
    compute_divergence_rhs(u_star_, v_star_, w_star_, mask, dt, rhs_);
    solve_poisson(p, rhs_, mask, bc_list);

    // 4. CORRECTOR STEP: Project trial velocity to divergence-free velocity field u^{n+1}
    apply_corrector(u_star_, v_star_, w_star_, p, mask, dt, u, v, w);
}

void NavierStokesOrchestrator::apply_pre_step_bcs(
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list,
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w,
    std::vector<double>& p
) {
    size_t nx = dims_.nx;
    size_t ny = dims_.ny;
    size_t nz = dims_.nz;

    // Force solid cells (mask == 0) velocity to zero
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t k = 0; k < nz; ++k) {
        for (size_t j = 0; j < ny; ++j) {
            for (size_t i = 0; i < nx; ++i) {
                size_t idx = i + nx * (j + ny * k);
                if (mask[idx] == 0) {
                    u[idx] = 0.0;
                    v[idx] = 0.0;
                    w[idx] = 0.0;
                }
            }
        }
    }

    // Apply boundary conditions for wall cells (mask == -1)
    for (const auto& bc : bc_list) {
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t k = 0; k < nz; ++k) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t i = 0; i < nx; ++i) {
                    size_t idx = i + nx * (j + ny * k);
                    if (mask[idx] != -1) continue;

                    // Match location specifier
                    bool matches_location = false;
                    if (bc.location == "x_min" && i == 0) matches_location = true;
                    else if (bc.location == "x_max" && i == nx - 1) matches_location = true;
                    else if (bc.location == "y_min" && j == 0) matches_location = true;
                    else if (bc.location == "y_max" && j == ny - 1) matches_location = true;
                    else if (bc.location == "z_min" && k == 0) matches_location = true;
                    else if (bc.location == "z_max" && k == nz - 1) matches_location = true;
                    else if (bc.location == "wall") matches_location = true;

                    if (matches_location) {
                        if (bc.type == "no-slip") {
                            u[idx] = 0.0;
                            v[idx] = 0.0;
                            w[idx] = 0.0;
                        } else if (bc.type == "inflow") {
                            u[idx] = bc.u_val;
                            v[idx] = bc.v_val;
                            w[idx] = bc.w_val;
                        } else if (bc.type == "pressure") {
                            p[idx] = bc.scalar_p;
                        }
                    }
                }
            }
        }
    }
}

void NavierStokesOrchestrator::compute_divergence_rhs(
    const std::vector<double>& u_star,
    const std::vector<double>& v_star,
    const std::vector<double>& w_star,
    const std::vector<int>& mask,
    double dt,
    std::vector<double>& rhs
) {
    size_t nx = dims_.nx;
    size_t ny = dims_.ny;
    size_t nz = dims_.nz;
    double idx2 = 0.5 / dims_.dx;
    double idy2 = 0.5 / dims_.dy;
    double idz2 = 0.5 / dims_.dz;
    double rho_dt = config_.density / dt;

    size_t ny_nz = ny * nz;
    size_t nz_val = nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t k = 1; k < nz - 1; ++k) {
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                size_t idx = i * ny_nz + j * nz_val + k;
                if (mask[idx] != 1) {
                    rhs[idx] = 0.0;
                    continue;
                }

                double dudx = (u_star[(i + 1) * ny_nz + j * nz_val + k] - u_star[(i - 1) * ny_nz + j * nz_val + k]) * idx2;
                double dvdy = (v_star[i * ny_nz + (j + 1) * nz_val + k] - v_star[i * ny_nz + (j - 1) * nz_val + k]) * idy2;
                double dwdz = (w_star[i * ny_nz + j * nz_val + (k + 1)] - w_star[i * ny_nz + j * nz_val + (k - 1)]) * idz2;

                rhs[idx] = rho_dt * (dudx + dvdy + dwdz);
            }
        }
    }
}

void NavierStokesOrchestrator::solve_poisson(
    std::vector<double>& p,
    const std::vector<double>& rhs,
    const std::vector<int>& mask,
    const std::vector<BoundaryCondition>& bc_list
) {
    size_t nx = dims_.nx;
    size_t ny = dims_.ny;
    size_t nz = dims_.nz;
    double idx2 = 1.0 / (dims_.dx * dims_.dx);
    double idy2 = 1.0 / (dims_.dy * dims_.dy);
    double idz2 = 1.0 / (dims_.dz * dims_.dz);
    double factor = 0.5 / (idx2 + idy2 + idz2);

    size_t ny_nz = ny * nz;
    size_t nz_val = nz;

    for (size_t iter = 0; iter < config_.max_poisson_iterations; ++iter) {
        
        // --- PASS 1: Update RED Cells ((i + j + k) % 2 == 0) ---
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t k = 1; k < nz - 1; ++k) {
            for (size_t j = 1; j < ny - 1; ++j) {
                for (size_t i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 != 0) continue;
                    size_t idx = i * ny_nz + j * nz_val + k;
                    if (mask[idx] != 1) continue;

                    double p_west  = p[(i - 1) * ny_nz + j * nz_val + k];
                    double p_east  = p[(i + 1) * ny_nz + j * nz_val + k];
                    double p_south = p[i * ny_nz + (j - 1) * nz_val + k];
                    double p_north = p[i * ny_nz + (j + 1) * nz_val + k];
                    double p_down  = p[i * ny_nz + j * nz_val + (k - 1)];
                    double p_up    = p[i * ny_nz + j * nz_val + (k + 1)];

                    p[idx] = factor * ((p_east + p_west) * idx2 +
                                       (p_north + p_south) * idy2 +
                                       (p_up + p_down) * idz2 - rhs[idx]);
                }
            }
        }

        // --- PASS 2: Update BLACK Cells ((i + j + k) % 2 == 1) ---
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t k = 1; k < nz - 1; ++k) {
            for (size_t j = 1; j < ny - 1; ++j) {
                for (size_t i = 1; i < nx - 1; ++i) {
                    if ((i + j + k) % 2 == 0) continue;
                    size_t idx = i * ny_nz + j * nz_val + k;
                    if (mask[idx] != 1) continue;

                    double p_west  = p[(i - 1) * ny_nz + j * nz_val + k];
                    double p_east  = p[(i + 1) * ny_nz + j * nz_val + k];
                    double p_south = p[i * ny_nz + (j - 1) * nz_val + k];
                    double p_north = p[i * ny_nz + (j + 1) * nz_val + k];
                    double p_down  = p[i * ny_nz + j * nz_val + (k - 1)];
                    double p_up    = p[i * ny_nz + j * nz_val + (k + 1)];

                    p[idx] = factor * ((p_east + p_west) * idx2 +
                                       (p_north + p_south) * idy2 +
                                       (p_up + p_down) * idz2 - rhs[idx]);
                }
            }
        }

        // --- PASS 3: Solid and Neumann Boundary Synchronization ---
        // 1. Solid cells (mask == 0) pressure Neumann zero-gradient averaging
        #pragma omp parallel for collapse(3) schedule(static)
        for (size_t k = 1; k < nz - 1; ++k) {
            for (size_t j = 1; j < ny - 1; ++j) {
                for (size_t i = 1; i < nx - 1; ++i) {
                    size_t idx = i * ny_nz + j * nz_val + k;
                    if (mask[idx] != 0) continue;

                    double sum = 0.0;
                    int count = 0;

                    auto add_if_fluid = [&](size_t n_idx) {
                        if (mask[n_idx] == 1) {
                            sum += p[n_idx];
                            count++;
                        }
                    };

                    add_if_fluid((i - 1) * ny_nz + j * nz_val + k);
                    add_if_fluid((i + 1) * ny_nz + j * nz_val + k);
                    add_if_fluid(i * ny_nz + (j - 1) * nz_val + k);
                    add_if_fluid(i * ny_nz + (j + 1) * nz_val + k);
                    add_if_fluid(i * ny_nz + j * nz_val + (k - 1));
                    add_if_fluid(i * ny_nz + j * nz_val + (k + 1));

                    if (count > 0) {
                        p[idx] = sum / count;
                    }
                }
            }
        }

        // 2. Wall Neumann pressure synchronization (mask == -1 without explicit pressure Dirichlet)
        for (const auto& bc : bc_list) {
            if (bc.type == "pressure") continue; // Keep Dirichlet fixed
            
            #pragma omp parallel for collapse(3) schedule(static)
            for (size_t k = 0; k < nz; ++k) {
                for (size_t j = 0; j < ny; ++j) {
                    for (size_t i = 0; i < nx; ++i) {
                        size_t idx = i * ny_nz + j * nz_val + k;
                        if (mask[idx] != -1) continue;

                        // Zero-gradient Neumann extrapolation from adjacent interior cell
                        size_t interior_idx = idx;
                        if (i == 0 && nx > 1) interior_idx = 1 * ny_nz + j * nz_val + k;
                        else if (i == nx - 1 && nx > 1) interior_idx = (nx - 2) * ny_nz + j * nz_val + k;
                        else if (j == 0 && ny > 1) interior_idx = i * ny_nz + 1 * nz_val + k;
                        else if (j == ny - 1 && ny > 1) interior_idx = i * ny_nz + (ny - 2) * nz_val + k;
                        else if (k == 0 && nz > 1) interior_idx = i * ny_nz + j * nz_val + 1;
                        else if (k == nz - 1 && nz > 1) interior_idx = i * ny_nz + j * nz_val + (nz - 2);

                        p[idx] = p[interior_idx];
                    }
                }
            }
        }
    }
}

void NavierStokesOrchestrator::apply_corrector(
    const std::vector<double>& u_star,
    const std::vector<double>& v_star,
    const std::vector<double>& w_star,
    const std::vector<double>& p,
    const std::vector<int>& mask,
    double dt,
    std::vector<double>& u,
    std::vector<double>& v,
    std::vector<double>& w
) {
    size_t nx = dims_.nx;
    size_t ny = dims_.ny;
    size_t nz = dims_.nz;
    double idx2 = 0.5 / dims_.dx;
    double idy2 = 0.5 / dims_.dy;
    double idz2 = 0.5 / dims_.dz;
    double dt_rho = dt / config_.density;

    size_t ny_nz = ny * nz;
    size_t nz_val = nz;

    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t k = 1; k < nz - 1; ++k) {
        for (size_t j = 1; j < ny - 1; ++j) {
            for (size_t i = 1; i < nx - 1; ++i) {
                size_t idx = i * ny_nz + j * nz_val + k;
                if (mask[idx] != 1) continue; // Active fluid only

                double dpdx = (p[(i + 1) * ny_nz + j * nz_val + k] - p[(i - 1) * ny_nz + j * nz_val + k]) * idx2;
                double dpdy = (p[i * ny_nz + (j + 1) * nz_val + k] - p[i * ny_nz + (j - 1) * nz_val + k]) * idy2;
                double dpdz = (p[i * ny_nz + j * nz_val + (k + 1)] - p[i * ny_nz + j * nz_val + (k - 1)]) * idz2;

                u[idx] = u_star[idx] - dt_rho * dpdx;
                v[idx] = v_star[idx] - dt_rho * dpdy;
                w[idx] = w_star[idx] - dt_rho * dpdz;
            }
        }
    }
}

} // namespace ops
