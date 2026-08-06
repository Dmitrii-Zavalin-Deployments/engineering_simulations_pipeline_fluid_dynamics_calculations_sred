#ifndef FIELD_SCHEMA_HPP
#define FIELD_SCHEMA_HPP

namespace navier_stokes {

enum class FI : int {
    // Primary Velocity Fields
    VX = 0,
    VY = 1,
    VZ = 2,
    
    // Intermediate Predictor Fields
    VX_STAR = 3,
    VY_STAR = 4,
    VZ_STAR = 5,
    
    // Pressure Fields
    P = 6,
    P_NEXT = 7,
    
    // Topological Mask
    MASK = 8,

    // Total number of fields for buffer allocation sizing
    NUM_FIELDS = 9
};

} // namespace navier_stokes

#endif // FIELD_SCHEMA_HPP
