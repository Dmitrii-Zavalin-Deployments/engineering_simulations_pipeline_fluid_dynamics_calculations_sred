#ifndef CELL_HPP
#define CELL_HPP

#include "field_schema.hpp"

namespace navier_stokes {

class CellView {
private:
    double* fields_buffer_;
    int index_;
    int nx_buf_;
    int ny_buf_;
    bool is_ghost_;

public:
    CellView(double* fields_buffer, int index, int nx_buf, int ny_buf, bool is_ghost = false)
        : fields_buffer_(fields_buffer), index_(index), nx_buf_(nx_buf), ny_buf_(ny_buf), is_ghost_(is_ghost) {}

    inline int index() const { return index_; }
    inline bool is_ghost() const { return is_ghost_; }

    inline double get_field(int field_id) const {
        return fields_buffer_[index_ * static_cast<int>(FI::NUM_FIELDS) + field_id];
    }

    inline void set_field(int field_id, double value) {
        fields_buffer_[index_ * static_cast<int>(FI::NUM_FIELDS) + field_id] = value;
    }

    // Topological Access
    inline int mask() const {
        return static_cast<int>(get_field(static_cast<int>(FI::MASK)));
    }
    
    inline void set_mask(int value) {
        set_field(static_cast<int>(FI::MASK), static_cast<double>(value));
    }

    // Physical Fields (Sovereign Inline Accessors)
    inline double vx() const { return get_field(static_cast<int>(FI::VX)); }
    inline void set_vx(double value) { set_field(static_cast<int>(FI::VX), value); }

    inline double vy() const { return get_field(static_cast<int>(FI::VY)); }
    inline void set_vy(double value) { set_field(static_cast<int>(FI::VY), value); }

    inline double vz() const { return get_field(static_cast<int>(FI::VZ)); }
    inline void set_vz(double value) { set_field(static_cast<int>(FI::VZ), value); }

    inline double vx_star() const { return get_field(static_cast<int>(FI::VX_STAR)); }
    inline void set_vx_star(double value) { set_field(static_cast<int>(FI::VX_STAR), value); }

    inline double vy_star() const { return get_field(static_cast<int>(FI::VY_STAR)); }
    inline void set_vy_star(double value) { set_field(static_cast<int>(FI::VY_STAR), value); }

    inline double vz_star() const { return get_field(static_cast<int>(FI::VZ_STAR)); }
    inline void set_vz_star(double value) { set_field(static_cast<int>(FI::VZ_STAR), value); }

    inline double p() const { return get_field(static_cast<int>(FI::P)); }
    inline void set_p(double value) { set_field(static_cast<int>(FI::P), value); }

    inline double p_next() const { return get_field(static_cast<int>(FI::P_NEXT)); }
    inline void set_p_next(double value) { set_field(static_cast<int>(FI::P_NEXT), value); }
};

} // namespace navier_stokes

#endif // CELL_HPP
