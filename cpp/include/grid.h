#ifndef GRID_H
#define GRID_H

#include <vector>

struct MacGrid {
    int nx, ny, nz;
    double dx, dy, dz;

    // Cell-centered fields
    std::vector<double> pressure; // size: nx * ny * nz
    std::vector<double> density;  // size: nx * ny * nz

    // Face-centered staggered velocities
    std::vector<double> u; // size: (nx + 1) * ny * nz (x-faces)
    std::vector<double> v; // size: nx * (ny + 1) * nz (y-faces)
    std::vector<double> w; // size: nx * ny * (nz + 1) (z-faces)

    // Constructor for convenient initialization
    MacGrid(int nx_, int ny_, int nz_, double dx_, double dy_, double dz_)
        : nx(nx_), ny(ny_), nz(nz_), dx(dx_), dy(dy_), dz(dz_),
          pressure(nx_ * ny_ * nz_, 0.0),
          density(nx_ * ny_ * nz_, 1.0),
          u((nx_ + 1) * ny_ * nz_, 0.0),
          v(nx_ * (ny_ + 1) * nz_, 0.0),
          w(nx_ * ny_ * (nz_ + 1), 0.0) {}
};

#endif // GRID_H
