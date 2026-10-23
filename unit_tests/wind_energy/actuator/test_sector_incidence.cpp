#include "ks_test_utils/MeshTest.H"

#include "src/wind_energy/actuator/ActParser.H"
#include "src/wind_energy/actuator/sector/ActuatorSector.H"
#include "src/wind_energy/actuator/sector/actuator_sector_ops.H"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>

#include "AMReX_ParallelDescriptor.H"
#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {
namespace {

namespace act = ::kynema_sgf::actuator;
using SectorData = act::ActuatorSector::DataType;
using Vector = ::kynema_sgf::vs::Vector;

class SectorIncidenceTest : public MeshTest
{
protected:
    static constexpr amrex::Real pi = 3.14159265358979323846_rt;
    static constexpr amrex::Real pitch = 20.0_rt;
    static constexpr amrex::Real tolerance =
        sizeof(amrex::Real) == sizeof(double) ? 1.0e-10_rt : 1.0e-3_rt;

    void populate_parameters() override
    {
        MeshTest::populate_parameters();
        amrex::ParmParse pp_amr("amr");
        pp_amr.add("max_level", 0);
        pp_amr.add("max_grid_size", 8);
        pp_amr.addarr("n_cell", amrex::Vector<int>{8, 8, 8});
        amrex::ParmParse pp_time("time");
        pp_time.add("fixed_dt", 1.0e-4_rt);
        amrex::ParmParse pp_geometry("geometry");
        pp_geometry.addarr(
            "prob_lo", amrex::Vector<amrex::Real>{-0.2_rt, -0.2_rt, -0.2_rt});
        pp_geometry.addarr(
            "prob_hi", amrex::Vector<amrex::Real>{0.2_rt, 0.2_rt, 0.2_rt});
    }

    void SetUp() override
    {
        MeshTest::SetUp();
        initialize_mesh();

        // Use a real parsed polar with an exactly linear lift segment over all
        // incidence angles exercised below, and constant positive profile drag.
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::ofstream file(m_airfoil);
            file << "! Sector incidence regression polar\n"
                 << "7 NumAlf\n"
                 << "! Alpha Cl Cd Cm\n"
                 << "! deg - - -\n"
                 << "-180 0 0.01 0\n"
                 << "-60 -3 0.01 0\n"
                 << "-20 -1 0.01 0\n"
                 << "0 0 0.01 0\n"
                 << "20 1 0.01 0\n"
                 << "60 3 0.01 0\n"
                 << "180 0 0.01 0\n";
            EXPECT_TRUE(file.good());
        }
        amrex::ParallelDescriptor::Barrier();

        amrex::ParmParse pp("Actuator.SectorIncidenceDefaults");
        pp.add("rotor_diameter", 0.08_rt);
        pp.add("root_radius_fraction", 0.2_rt);
        pp.add("num_blades", 2);
        pp.addarr(
            "epsilon",
            amrex::Vector<amrex::Real>{0.006_rt, 0.003_rt, 0.004_rt});
        pp.add("epsilon_dr", 1.5_rt);
        pp.add("epsilon_dl", 1.5_rt);
        pp.add("min_chord_dr", 2.0_rt);
        pp.add("airfoil_table", m_airfoil);
        pp.add("airfoil_type", std::string("openfast"));
        pp.add("gaussian_type", std::string("direct"));
        pp.addarr("span_locs", amrex::Vector<amrex::Real>{0.0_rt, 1.0_rt});
        pp.addarr("chord", amrex::Vector<amrex::Real>{0.008_rt, 0.008_rt});
    }

    void TearDown() override
    {
        amrex::ParallelDescriptor::Barrier();
        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::remove(m_airfoil.c_str());
        }
        MeshTest::TearDown();
    }

    std::unique_ptr<SectorData> make_sector(const amrex::Real spin)
    {
        const std::string label =
            spin > 0.0_rt ? "IncidencePositive" : "IncidenceNegative";
        amrex::ParmParse pp("Actuator." + label);
        pp.add("omega", spin * 200.0_rt);
        pp.addarr(
            "twist", amrex::Vector<amrex::Real>{spin * pitch, spin * pitch});
        pp.addarr("center", amrex::Vector<amrex::Real>{0.0_rt, 0.0_rt, 0.0_rt});
        pp.addarr(
            "rotor_normal", amrex::Vector<amrex::Real>{0.0_rt, 0.0_rt, 1.0_rt});
        auto data =
            std::make_unique<SectorData>(sim(), label, spin > 0.0_rt ? 0 : 1);
        const act::utils::ActParser parser(
            "Actuator.SectorIncidenceDefaults", "Actuator." + label);
        act::ops::ReadInputsOp<act::ActuatorSector, act::ActSrcSector>()(
            *data, parser);
        act::ops::InitDataOp<act::ActuatorSector, act::ActSrcSector>()(*data);
        return data;
    }

    void apply_axial_flow(SectorData& data, const amrex::Real axial_velocity)
    {
        act::ops::UpdatePosOp<act::ActuatorSector, act::ActSrcSector>()(data);
        for (auto& velocity : data.grid().vel) {
            velocity = {0.0_rt, 0.0_rt, axial_velocity};
        }
        act::ops::ComputeForceOp<act::ActuatorSector, act::ActSrcSector>()(
            data);
    }

    const std::string m_airfoil = "sector_incidence_regression_airfoil.txt";
};

} // namespace

TEST_F(SectorIncidenceTest, zero_axial_flow_recovers_positive_pitch)
{
    for (const amrex::Real spin : {-1.0_rt, 1.0_rt}) {
        auto data = make_sector(spin);
        ASSERT_GT(data->meta().radius.size(), 1U);
        apply_axial_flow(*data, 0.0_rt);
        for (const auto alpha : data->meta().aoa) {
            EXPECT_NEAR(alpha, pitch, tolerance);
        }
        for (const auto cl : data->meta().cl) {
            EXPECT_NEAR(cl, 1.0_rt, tolerance);
        }
        EXPECT_GT(data->meta().thrust, 0.0_rt);
        EXPECT_GT(spin * data->meta().torque, 0.0_rt);
    }
}

TEST_F(SectorIncidenceTest, downwash_reduces_incidence_and_upflow_increases_it)
{
    for (const amrex::Real spin : {-1.0_rt, 1.0_rt}) {
        auto data = make_sector(spin);
        const auto& meta = data->meta();
        const int nr = static_cast<int>(meta.radius.size());
        for (const amrex::Real axial_velocity : {-0.3_rt, 0.3_rt}) {
            apply_axial_flow(*data, axial_velocity);
            for (int blade = 0; blade < meta.num_blades; ++blade) {
                for (int ir = 0; ir < nr; ++ir) {
                    const int index = blade * nr + ir;
                    const amrex::Real inflow_degrees =
                        std::atan2(axial_velocity, 200.0_rt * meta.radius[ir]) *
                        180.0_rt / pi;
                    const amrex::Real expected = pitch - inflow_degrees;
                    EXPECT_NEAR(meta.aoa[index], expected, tolerance);
                    EXPECT_NEAR(meta.cl[index], 0.05_rt * expected, tolerance);
                    if (axial_velocity > 0.0_rt) {
                        EXPECT_LT(meta.aoa[index], pitch);
                    } else {
                        EXPECT_GT(meta.aoa[index], pitch);
                    }
                }
            }
        }
    }
}

TEST_F(
    SectorIncidenceTest, chord_aligned_flow_has_zero_incidence_at_every_station)
{
    for (const amrex::Real spin : {-1.0_rt, 1.0_rt}) {
        auto data = make_sector(spin);
        auto& meta = data->meta();
        auto& grid = data->grid();
        act::ops::UpdatePosOp<act::ActuatorSector, act::ActSrcSector>()(*data);
        const int nr = static_cast<int>(meta.radius.size());
        for (int blade = 0; blade < meta.num_blades; ++blade) {
            for (int ir = 0; ir < nr; ++ir) {
                // Physical incoming flow is parallel to the leading-edge
                // chord when W/(|omega|r) = tan(pitch): incidence is zero.
                grid.vel[blade * nr + ir] = {
                    0.0_rt, 0.0_rt,
                    200.0_rt * meta.radius[ir] *
                        std::tan(pitch * pi / 180.0_rt)};
            }
        }
        act::ops::ComputeForceOp<act::ActuatorSector, act::ActSrcSector>()(
            *data);
        for (const auto alpha : meta.aoa) {
            EXPECT_NEAR(alpha, 0.0_rt, tolerance);
        }
        for (const auto cl : meta.cl) {
            EXPECT_NEAR(cl, 0.0_rt, tolerance);
        }
    }
}

TEST_F(
    SectorIncidenceTest,
    mirrored_loads_torque_and_projected_chord_are_consistent)
{
    auto positive = make_sector(1.0_rt);
    auto negative = make_sector(-1.0_rt);
    apply_axial_flow(*positive, 0.3_rt);
    apply_axial_flow(*negative, 0.3_rt);
    const auto& positive_meta = positive->meta();
    const auto& negative_meta = negative->meta();
    ASSERT_EQ(positive_meta.aoa.size(), negative_meta.aoa.size());
    EXPECT_NEAR(positive_meta.thrust, negative_meta.thrust, tolerance);
    EXPECT_NEAR(positive_meta.torque, -negative_meta.torque, tolerance);
    EXPECT_GT(positive_meta.thrust, 0.0_rt);
    EXPECT_GT(positive_meta.torque, 0.0_rt);
    EXPECT_LT(negative_meta.torque, 0.0_rt);
    for (std::size_t index = 0; index < positive_meta.aoa.size(); ++index) {
        EXPECT_NEAR(
            positive_meta.aoa[index], negative_meta.aoa[index], tolerance);
        EXPECT_NEAR(
            positive_meta.blade_force_normal[index],
            negative_meta.blade_force_normal[index], tolerance);
        EXPECT_NEAR(
            positive_meta.blade_force_theta[index],
            -negative_meta.blade_force_theta[index], tolerance);
    }

    for (const auto* data : {positive.get(), negative.get()}) {
        const auto& meta = data->meta();
        const auto& grid = data->grid();
        const amrex::Real spin = meta.omega > 0.0_rt ? 1.0_rt : -1.0_rt;
        const Vector normal{0.0_rt, 0.0_rt, 1.0_rt};
        for (std::size_t index = 0; index < grid.pos.size(); ++index) {
            // Recover the local basis from actual projected coordinates,
            // independently of the implementation's blade_basis routine.
            const Vector radial = (grid.pos[index] - meta.center).unit();
            const Vector tangential = -(normal ^ radial);
            const Vector physical_leading_edge =
                spin * std::cos(pitch * pi / 180.0_rt) * tangential -
                std::sin(pitch * pi / 180.0_rt) * normal;
            const auto local_chord =
                grid.orientation[index] & physical_leading_edge;
            EXPECT_NEAR(std::abs(local_chord.x()), 1.0_rt, tolerance);
            EXPECT_NEAR(local_chord.y(), 0.0_rt, tolerance);
            EXPECT_NEAR(local_chord.z(), 0.0_rt, tolerance);
        }
        EXPECT_NEAR(meta.integrated_force.z(), meta.thrust, tolerance);
        EXPECT_NEAR(meta.integrated_moment.z(), -meta.torque, tolerance);
    }
}

} // namespace kynema_sgf_tests
