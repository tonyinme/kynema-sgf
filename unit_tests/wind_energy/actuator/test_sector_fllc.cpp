#include "src/wind_energy/actuator/sector/sector_fllc.H"
#include "AMReX.H"
#include "gtest/gtest.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

using namespace amrex::literals;
using namespace kynema_sgf;
using namespace kynema_sgf::actuator;
using sector::SectorFLLCBlade;

namespace {
int assertions = 0;
using Real = amrex::Real;
const RealList radius{0.0035_rt, 0.00475_rt, 0.0065_rt, 0.008_rt, 0.00875_rt};
const RealList widths{0.001_rt, 0.0015_rt, 0.002_rt, 0.001_rt, 0.0005_rt};
const RealList chord{0.005_rt, 0.006_rt, 0.007_rt, 0.004_rt, 0.002_rt};
const Real physical_epsilon = 0.006_rt;
const Real target = 0.25_rt;

void require(bool ok, const std::string& label)
{
    ++assertions;
    if (!ok) {
        throw std::runtime_error(label);
    }
}

void close(
    Real got,
    Real expected,
    const std::string& label,
    Real relative = 3.0e-12_rt,
    Real absolute = 3.0e-12_rt)
{
    require(
        std::isfinite(got) && std::isfinite(expected) &&
            std::abs(got - expected) <=
                absolute +
                    relative * std::max(std::abs(got), std::abs(expected)),
        label + ": got=" + std::to_string(got) +
            " expected=" + std::to_string(expected));
}

void close_vector(
    const vs::Vector& got,
    const vs::Vector& expected,
    const std::string& label,
    Real tolerance = 3.0e-12_rt)
{
    for (int k = 0; k < AMREX_SPACEDIM; ++k) {
        close(
            got[k], expected[k], label + " component=" + std::to_string(k),
            tolerance, tolerance);
    }
}

SectorFLLCBlade make_blade(Real relaxation = 1.0_rt)
{
    SectorFLLCBlade blade;
    blade.initialize(
        radius, widths, chord, physical_epsilon, target, relaxation);
    return blade;
}

void load(SectorFLLCBlade& blade, bool mirrored = false, bool add_drag = false)
{
    const Real sign = mirrored ? -1.0_rt : 1.0_rt;
    for (int j = 0; j < static_cast<int>(radius.size()); ++j) {
        const auto velocity = vs::Vector{
            sign * (-15.0_rt - 5.0_rt * static_cast<Real>(j)),
            1.0_rt + 0.3_rt * static_cast<Real>(j), 0.0_rt};
        // Force perpendicular to relative wind, with a normal component
        // pointing with positive propeller downwash. Mirror theta only.
        auto force = vs::Vector{
            sign * (1.0_rt + 0.3_rt * static_cast<Real>(j)),
            15.0_rt + 5.0_rt * static_cast<Real>(j), 0.0_rt};
        force *= 0.2_rt + 0.025_rt * static_cast<Real>(j);
        if (add_drag) {
            force =
                force - velocity * (0.17_rt + 0.01_rt * static_cast<Real>(j));
        }
        blade.store(j, force, velocity);
    }
}

Real kernel(Real distance, Real epsilon)
{
    const Real e2 = epsilon * epsilon;
    if (distance == 0.0_rt) {
        return 0.5_rt / e2;
    }
    const Real r2 = distance * distance;
    const Real exponential = std::exp(-r2 / e2);
    return exponential / e2 + (exponential - 1.0_rt) / (2.0_rt * r2);
}

// Independent direct sum of the published variable-chord integral, using
// per-unit-span inputs recovered from the actual sector cell weights.
VecList direct_oracle(const SectorFLLCBlade& blade)
{
    VecList result(radius.size(), vs::Vector::zero());
    for (amrex::Long i = 0; i < radius.size(); ++i) {
        for (amrex::Long j = 0; j < radius.size(); ++j) {
            const auto velocity = blade.relative_velocity[j];
            const auto force_per_span = blade.force[j] / widths[j];
            const Real speed = vs::mag(velocity);
            require(speed > 0.0_rt, "oracle positive speed");
            const auto lift =
                force_per_span -
                velocity * ((force_per_span & velocity) / (speed * speed));
            const Real distance = std::abs(radius[i] - radius[j]);
            const Real delta_kernel = kernel(distance, target * chord[j]) -
                                      kernel(distance, blade.applied_epsilon[j]);
            result[i] = result[i] + lift / speed *
                                        (delta_kernel * widths[j] /
                                         (2.0_rt * std::numbers::pi_v<Real>));
        }
    }
    return result;
}

} // namespace

TEST(SectorFLLCTest, fresh_blade_has_no_correction_or_previous_loads)
{
    auto blade = make_blade();
    require(!blade.have_loads, "fresh history");
    blade.update();
    for (const auto& value : blade.state.correction_velocity) {
        close_vector(value, vs::Vector::zero(), "fresh correction");
    }
}

TEST(
    SectorFLLCTest,
    equal_physical_and_target_epsilon_gives_exactly_zero_correction)
{
    SectorFLLCBlade blade;
    blade.initialize(
        radius, widths, RealList(radius.size(), 0.024_rt), physical_epsilon,
        target, 0.1_rt);
    load(blade, false, true);
    for (int iteration = 0; iteration < 4; ++iteration) {
        blade.update();
        for (const auto& value : blade.state.correction_velocity) {
            close_vector(value, vs::Vector::zero(), "equal epsilon");
        }
    }
}

TEST(
    SectorFLLCTest,
    midpoint_cell_weights_and_force_units_include_both_endpoints)
{
    auto blade = make_blade();
    load(blade);
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        close(blade.state.dr[j], widths[j], "actual cell width");
        const auto velocity = blade.relative_velocity[j];
        const auto expected =
            vs::Vector{
                1.0_rt + 0.3_rt * static_cast<Real>(j),
                15.0_rt + 5.0_rt * static_cast<Real>(j), 0.0_rt} *
            (0.2_rt + 0.025_rt * static_cast<Real>(j));
        close_vector(
            blade.force[j], expected * widths[j], "integrated cell force");
        close(expected & velocity, 0.0_rt, "perpendicular sample force");
    }
    const auto expected = direct_oracle(blade);
    blade.update();
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        close_vector(
            blade.state.lift[j], blade.force[j] / widths[j],
            "recovered per-unit-span lift");
        close_vector(
            blade.state.correction_velocity[j], expected[j],
            "independent direct integral", 2.0e-11_rt);
    }
}

TEST(SectorFLLCTest, mirror_theta_leaves_normal_correction_invariant)
{
    auto positive = make_blade();
    auto negative = make_blade();
    load(positive, false, true);
    load(negative, true, true);
    positive.update();
    negative.update();
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        const auto value = positive.state.correction_velocity[j];
        close_vector(
            negative.state.correction_velocity[j],
            {-value.x(), value.y(), value.z()}, "spin mirror");
    }
}

TEST(
    SectorFLLCTest,
    force_parallel_to_relative_velocity_does_not_affect_correction)
{
    auto lift = make_blade();
    auto lift_and_drag = make_blade();
    load(lift);
    load(lift_and_drag, false, true);
    lift.update();
    lift_and_drag.update();
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        close_vector(
            lift.state.correction_velocity[j],
            lift_and_drag.state.correction_velocity[j],
            "drag removed by projection");
    }
    auto drag_only = make_blade();
    for (int j = 0; j < static_cast<int>(radius.size()); ++j) {
        const vs::Vector v{-20.0_rt, 2.0_rt, 0.0_rt};
        drag_only.store(j, -3.0_rt * v, v);
    }
    drag_only.update();
    for (const auto& value : drag_only.state.correction_velocity) {
        close_vector(value, vs::Vector::zero(), "pure drag zero correction");
    }
}

TEST(SectorFLLCTest, zero_speed_and_zero_force_remain_finite_and_zero)
{
    auto blade = make_blade();
    for (int j = 0; j < static_cast<int>(radius.size()); ++j) {
        blade.store(j, vs::Vector::zero(), vs::Vector::zero());
    }
    for (int iteration = 0; iteration < 4; ++iteration) {
        blade.update();
        for (const auto& value : blade.state.correction_velocity) {
            close_vector(value, vs::Vector::zero(), "zero speed");
        }
    }
}

TEST(SectorFLLCTest, relaxation_follows_history_and_scratch_velocity_is_reset)
{
    auto direct = make_blade();
    auto relaxed = make_blade(0.2_rt);
    load(direct);
    load(relaxed);
    direct.update();
    relaxed.update();
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        close_vector(
            relaxed.state.correction_velocity[j],
            0.2_rt * direct.state.correction_velocity[j], "first relaxation");
    }
    relaxed.update();
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        close_vector(
            relaxed.state.correction_velocity[j],
            0.36_rt * direct.state.correction_velocity[j], "second relaxation");
        close_vector(
            relaxed.velocity[j], relaxed.state.correction_velocity[j],
            "scratch velocity contains current correction only");
    }
}

TEST(SectorFLLCTest, blades_retain_independent_histories)
{
    auto first = make_blade(0.2_rt);
    auto second = make_blade(0.2_rt);
    load(first);
    load(second);
    first.update();
    second.update();
    const auto saved = second.state.correction_velocity;
    first.update();
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        close_vector(
            second.state.correction_velocity[j], saved[j],
            "unmodified other blade");
        close_vector(
            first.state.correction_velocity[j], 1.8_rt * saved[j],
            "independent relaxation age");
    }
}

TEST(
    SectorFLLCTest,
    isotropic_integral_is_invariant_under_rigid_frame_transforms)
{
    auto original = make_blade();
    auto transformed = make_blade();
    load(original, false, true);
    load(transformed, false, true);
    // A proper 90-degree rotation and a translation of the distance
    // geometry must not affect the scalar spanwise kernel.
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        transformed.position[j] = {0.2_rt, -0.4_rt + radius[j], 0.7_rt};
        const auto force = transformed.force[j];
        const auto velocity = transformed.relative_velocity[j];
        transformed.force[j] = {-force.y(), force.x(), force.z()};
        transformed.relative_velocity[j] = {
            -velocity.y(), velocity.x(), velocity.z()};
    }
    original.update();
    transformed.update();
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        const auto value = original.state.correction_velocity[j];
        close_vector(
            transformed.state.correction_velocity[j],
            {-value.y(), value.x(), value.z()}, "rigid transform", 3.0e-10_rt);
    }
}

TEST(SectorFLLCTest, finite_loaded_span_adds_downwash_toward_its_tip)
{
    constexpr int count = 81;
    const Real dr = 0.00025_rt;
    RealList r(count), w(count, dr), c(count, 0.005_rt);
    for (int j = 0; j < count; ++j) {
        r[j] = 0.003_rt + (static_cast<Real>(j) + 0.5_rt) * dr;
    }
    SectorFLLCBlade blade;
    blade.initialize(r, w, c, physical_epsilon, target, 1.0_rt);
    for (int j = 0; j < count; ++j) {
        blade.store(j, {0.0_rt, 5.0_rt, 0.0_rt}, {-20.0_rt, 0.0_rt, 0.0_rt});
    }
    blade.update();
    const auto tip = blade.state.correction_velocity.back();
    const auto center = blade.state.correction_velocity[count / 2];
    require(tip.y() > 0.0_rt, "positive tip downwash");
    require(
        tip.y() > center.y(), "finite-span edge correction exceeds interior");
    close(tip.x(), 0.0_rt, "no tangential correction for normal lift");
}

TEST(SectorFLLCTest, adapter_exactly_matches_existing_wing_fllc_operator)
{
    auto adapter = make_blade(0.1_rt);
    auto reference = make_blade(0.1_rt);
    load(adapter, false, true);
    load(reference, false, true);
    for (int iteration = 0; iteration < 5; ++iteration) {
        adapter.update();
        std::fill(
            reference.velocity.begin(), reference.velocity.end(),
            vs::Vector::zero());
        auto view = reference.view();
        FLLCOp{}(view, reference.state);
        for (amrex::Long j = 0; j < radius.size(); ++j) {
            close_vector(
                adapter.state.correction_velocity[j],
                reference.state.correction_velocity[j],
                "existing FLLC equivalence", 0.0_rt);
        }
    }
}

TEST(SectorFLLCTest, varying_applied_width_matches_source_station_integral)
{
    RealList profile;
    for (const auto c : chord) profile.push_back(std::max(0.0015_rt, target * c));
    SectorFLLCBlade blade;
    blade.initialize(radius, widths, chord, profile, target, 1.0_rt);
    load(blade, false, true);
    const auto expected = direct_oracle(blade);
    blade.update();
    for (amrex::Long i = 0; i < radius.size(); ++i) {
        EXPECT_DOUBLE_EQ(blade.applied_epsilon[i], profile[i]);
        close_vector(blade.state.correction_velocity[i], expected[i],
                     "source-width direct integral", 2.0e-11_rt);
    }
    // Nonlocal floor contributions can affect a station whose own applied
    // width already equals its target. A local-width-only correction is wrong.
    EXPECT_DOUBLE_EQ(profile[2], target * chord[2]);
    EXPECT_GT(vs::mag(blade.state.correction_velocity[2]), 1.0e-8_rt);
}

TEST(SectorFLLCTest, varying_equal_applied_and_target_widths_cancel_exactly)
{
    RealList profile;
    for (const auto c : chord) profile.push_back(target * c);
    SectorFLLCBlade blade;
    blade.initialize(radius, widths, chord, profile, target, 0.1_rt);
    load(blade, false, true);
    for (int step = 0; step < 5; ++step) {
        blade.update();
        for (const auto& v : blade.state.correction_velocity)
            EXPECT_DOUBLE_EQ(vs::mag(v), 0.0_rt);
    }
}

TEST(SectorFLLCTest, applied_width_is_source_index_not_observation_index)
{
    RealList profile;
    for (const auto c : chord) profile.push_back(std::max(0.0015_rt, target * c));
    SectorFLLCBlade blade;
    blade.initialize(radius, widths, chord, profile, target, 1.0_rt);
    load(blade);
    // Only station 2 carries lift; its two widths agree. Its contribution
    // must cancel at EVERY observer, including observers with other widths.
    for (amrex::Long j = 0; j < radius.size(); ++j)
        if (j != 2) blade.force[j] = vs::Vector::zero();
    blade.update();
    for (const auto& v : blade.state.correction_velocity)
        EXPECT_DOUBLE_EQ(vs::mag(v), 0.0_rt);
}

TEST(SectorFLLCTest, scalar_and_constant_profile_keep_legacy_result_exactly)
{
    auto scalar = make_blade(0.1_rt);
    SectorFLLCBlade profile;
    profile.initialize(radius, widths, chord,
                       RealList(radius.size(), physical_epsilon), target, 0.1_rt);
    load(scalar, false, true);
    load(profile, false, true);
    for (int step = 0; step < 5; ++step) {
        scalar.update();
        profile.update();
        for (amrex::Long j = 0; j < radius.size(); ++j)
            close_vector(scalar.state.correction_velocity[j],
                         profile.state.correction_velocity[j],
                         "scalar/profile backward compatibility", 0.0_rt);
    }
}

TEST(
    SectorFLLCTest,
    upright_negative_normal_fluid_lift_preserves_wing_fllc_under_frame_reflection)
{
    auto old_frame = make_blade(0.1_rt);
    auto upright = make_blade(0.1_rt);
    load(old_frame, false, true);
    load(upright, false, true);
    for (amrex::Long j = 0; j < radius.size(); ++j) {
        const auto f = upright.force[j];
        const auto v = upright.relative_velocity[j];
        upright.force[j] = {f.x(), -f.y(), f.z()};
        upright.relative_velocity[j] = {v.x(), -v.y(), v.z()};
    }
    for (int iteration = 0; iteration < 5; ++iteration) {
        old_frame.update();
        upright.update();
        for (amrex::Long j = 0; j < radius.size(); ++j) {
            const auto old_du = old_frame.state.correction_velocity[j];
            close_vector(
                upright.state.correction_velocity[j],
                {old_du.x(), -old_du.y(), old_du.z()},
                "normal reflection with upright fluid force");
        }
    }
    const auto oracle = direct_oracle(upright);
    for (const auto& v : oracle)
        require(std::isfinite(v.y()), "finite upright correction");
}
