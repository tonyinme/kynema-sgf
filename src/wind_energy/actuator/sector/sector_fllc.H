#ifndef SECTOR_FLLC_H
#define SECTOR_FLLC_H

#include "src/wind_energy/actuator/FLLCOp.H"
#include "AMReX_BLProfiler.H"
#include <algorithm>
#include <cmath>

namespace kynema_sgf::actuator::sector {

/** Experimental adapter to the existing wing FLLC, one independent blade.
 *
 * All velocities and fluid forces use the co-rotating (theta, normal, radial)
 * component frame. Canonical positions supply only spanwise distances. The
 * previous corrected relative velocity and fluid load drive the next relaxed
 * update, as in the wing implementation. No swept projection points enter FLLC.
 */
struct SectorFLLCBlade
{
    FLLCData state;
    VecList position;
    VecList force;
    VecList velocity;
    VecList relative_velocity;
    RealList chord;
    RealList applied_epsilon;
    bool have_loads{false};

    ComponentView view()
    {
        ComponentView result;
        result.pos = ::kynema_sgf::utils::slice(position, 0);
        result.vel_pos = ::kynema_sgf::utils::slice(position, 0);
        result.force = ::kynema_sgf::utils::slice(force, 0);
        result.vel = ::kynema_sgf::utils::slice(velocity, 0);
        result.vel_rel = ::kynema_sgf::utils::slice(relative_velocity, 0);
        result.chord = ::kynema_sgf::utils::slice(chord, 0);
        return result;
    }

    void initialize(
        const RealList& radius,
        const RealList& radial_weights,
        const RealList& local_chord,
        const amrex::Real physical_epsilon,
        const amrex::Real target_epsilon_chord,
        const amrex::Real relaxation)
    {
        initialize(
            radius, radial_weights, local_chord,
            RealList(radius.size(), physical_epsilon), target_epsilon_chord,
            relaxation);
    }

    void initialize(
        const RealList& radius,
        const RealList& radial_weights,
        const RealList& local_chord,
        const RealList& physical_epsilon,
        const amrex::Real target_epsilon_chord,
        const amrex::Real relaxation)
    {
        const auto n = radius.size();
        AMREX_ALWAYS_ASSERT(
            n >= 2 && radial_weights.size() == n && local_chord.size() == n &&
            physical_epsilon.size() == n);
        for (const auto width : physical_epsilon) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::isfinite(width) && width > 0.0_rt,
                "Sector FLLC applied widths must be finite and positive");
        }
        state = FLLCData{};
        applied_epsilon = physical_epsilon;
        // Retain the scalar for legacy diagnostics; the rotor operator below
        // receives every actual per-station width, including a grid floor.
        state.epsilon = applied_epsilon.front();
        state.relaxation_factor = relaxation;
        state.nonuniform = false;
        chord = local_chord;
        position.resize(n);
        force.assign(n, vs::Vector::zero());
        velocity.assign(n, vs::Vector::zero());
        relative_velocity.assign(n, vs::Vector::zero());
        for (amrex::Long i = 0; i < n; ++i) {
            position[i] = {radius[i], 0.0_rt, 0.0_rt};
        }
        auto component = view();
        fllc_init(state, component, target_epsilon_chord);
        // Wing nodes have half endpoint weights; sector points are cell
        // midpoints. Use the actual sector weights at ALL stations, including
        // endpoints, both for integrated force and induced-velocity quadrature.
        state.dr = radial_weights;
        have_loads = false;
    }

    void update()
    {
        BL_PROFILE("ActuatorSector::FLLC");
        if (!have_loads) {
            return;
        }
        // FLLCOp adds du to this scratch velocity. The raw CFD velocity is
        // never modified; the caller adds du in the current section frame.
        std::fill(velocity.begin(), velocity.end(), vs::Vector::zero());
        auto component = view();
        FLLCOp{}(component, state, &applied_epsilon);
    }

    void store(
        const int station,
        const vs::Vector& fluid_force_per_span,
        const vs::Vector& corrected_relative_velocity)
    {
        force[station] = fluid_force_per_span * state.dr[station];
        relative_velocity[station] = corrected_relative_velocity;
        have_loads = true;
    }
};

} // namespace kynema_sgf::actuator::sector

#endif
