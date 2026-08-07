#ifndef SCALING_HPP
#define SCALING_HPP

#include "base_operator.hpp"

/**
 * Returns the scaling factor (dt / rho) for the Predictor and Corrector Steps.
 * Guards against vacuum density (rho <= 0) and non-finite results.
 */
double get_dt_over_rho(double dt, double rho);

/**
 * Returns the scaling factor (rho / dt) for the Pressure Poisson Equation.
 * Guards against zero or negative time-step (dt <= 0) and non-finite results.
 */
double get_rho_over_dt(double dt, double rho);

#endif // SCALING_HPP
