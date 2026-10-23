#include "ks_test_utils/MeshTest.H"

#include "src/wind_energy/actuator/Actuator.H"
#include "src/wind_energy/actuator/ActuatorContainer.H"
#include "src/wind_energy/actuator/ActuatorModel.H"
#include "src/wind_energy/actuator/ActParser.H"
#include "src/wind_energy/actuator/sector/ActuatorSector.H"
#include "src/wind_energy/actuator/sector/actuator_sector_ops.H"
#include "src/utilities/constants.H"
#include "src/utilities/tagging/ActuatorRefinement.H"

#include <fstream>
#include <limits>

#include "AMReX_REAL.H"

using namespace amrex::literals;

namespace kynema_sgf_tests {
namespace {

class ActuatorSectorTest : public MeshTest
{
protected:
    void populate_parameters() override
    {
        MeshTest::populate_parameters();

        {
            amrex::ParmParse pp("amr");
            amrex::Vector<int> ncell{{16, 16, 16}};
            pp.add("max_level", 0);
            pp.add("max_grid_size", 16);
            pp.addarr("n_cell", ncell);
        }
        {
            amrex::ParmParse pp("time");
            pp.add("fixed_dt", 1.0e-4_rt);
        }
        {
            amrex::ParmParse pp("geometry");
            pp.addarr(
                "prob_lo",
                amrex::Vector<amrex::Real>{-0.2_rt, -0.2_rt, -0.2_rt});
            pp.addarr(
                "prob_hi", amrex::Vector<amrex::Real>{0.2_rt, 0.2_rt, 0.2_rt});
        }
    }

    void initialize_domain()
    {
        initialize_mesh();
        sim().repo().declare_field("actuator_src_term", 3, 0);
        auto& vel = sim().repo().declare_field("velocity", 3, 3);
        auto& density = sim().repo().declare_field("density", 1, 3);
        vel.setVal(0.0_rt);
        density.setVal(1.0_rt);
        kynema_sgf::actuator::ActuatorContainer::ParticleType::NextID(1U);
    }

    void populate_actuator_inputs(
        const bool use_epsilon = false,
        const amrex::Real epsilon = 0.0_rt,
        const bool use_epsilon_chord = true,
        const amrex::Real epsilon_chord = 0.25_rt,
        const amrex::Real epsilon_min = 0.0_rt)
    {
        amrex::ParmParse pp_a("Actuator");
        pp_a.add("labels", (std::string) "R1");
        pp_a.add("type", (std::string) "ActuatorSector");

        amrex::ParmParse pp("Actuator.ActuatorSector");
        pp.add("rotor_diameter", 0.1_rt);
        pp.add("root_radius_fraction", 0.18_rt);
        pp.add("num_blades", 2);
        pp.add("omega", 2500.0_rt);
        if (use_epsilon) {
            pp.add("epsilon", epsilon);
        }
        if (use_epsilon_chord) {
            pp.add("epsilon_chord", epsilon_chord);
            pp.add("epsilon_min", epsilon_min);
        }
        pp.add("epsilon_dr", 1.5_rt);
        pp.add("epsilon_dl", 1.5_rt);
        pp.add("min_chord_dr", 2.0_rt);
        pp.add("airfoil_table", m_afname);
        pp.add("airfoil_type", (std::string) "openfast");
        pp.add("gaussian_type", (std::string) "table");
        pp.addarr("span_locs", amrex::Vector<amrex::Real>{0.0_rt, 1.0_rt});
        pp.addarr("chord", amrex::Vector<amrex::Real>{0.01_rt, 0.006_rt});
        pp.addarr("twist", amrex::Vector<amrex::Real>{12.0_rt, 4.0_rt});

        amrex::ParmParse pp_r("Actuator.R1");
        pp_r.addarr(
            "center", amrex::Vector<amrex::Real>{0.0_rt, 0.0_rt, 0.0_rt});
        pp_r.addarr(
            "translation_velocity",
            amrex::Vector<amrex::Real>{0.0_rt, 0.0_rt, 1.0_rt});
        pp_r.addarr(
            "rotor_normal", amrex::Vector<amrex::Real>{1.0_rt, 0.0_rt, 0.0_rt});
    }

    const std::string m_afname = "sector_airfoil.txt";
};

void write_airfoil_file(const std::string& fname)
{
    std::ofstream os(fname);
    os << "! OpenFAST-style test polar\n";
    os << "        5   NumAlf\n";
    os << "! Alpha Cl Cd Cm\n";
    os << "! deg   -  -  -\n";
    os << " -180.0  0.0 0.04 0.0\n";
    os << "  -10.0 -0.5 0.02 0.0\n";
    os << "    0.0  0.0 0.01 0.0\n";
    os << "   10.0  0.5 0.02 0.0\n";
    os << "  180.0  0.0 0.04 0.0\n";
}

void remove_file(const std::string& fname)
{
    std::ifstream f(fname);
    if (f.good()) {
        remove(fname.c_str());
    }
}

class ActPhysicsTest : public ::kynema_sgf::actuator::Actuator
{
public:
    explicit ActPhysicsTest(::kynema_sgf::CFDSim& sim)
        : ::kynema_sgf::actuator::Actuator(sim)
    {}

protected:
    void prepare_outputs() override {}
};

class ActuatorSectorRefinementTest : public ActuatorSectorTest
{
public:
    void populate_parameters() override
    {
        ActuatorSectorTest::populate_parameters();
        amrex::ParmParse pp("amr");
        pp.remove("max_level");
        pp.add("max_level", 1);
    }

    void create_mesh_instance() override
    {
        if (!m_mesh) {
            reset_prob_domain();
            m_mesh = std::make_unique<RefineMesh>();
        }
    }
};

} // namespace

TEST_F(ActuatorSectorTest, act_model_init)
{
    write_airfoil_file(m_afname);
    initialize_domain();
    populate_actuator_inputs();

    ::kynema_sgf::actuator::ActModel<
        ::kynema_sgf::actuator::ActuatorSector,
        ::kynema_sgf::actuator::ActSrcSector>
        sector(sim(), "R1", 0);
    {
        ::kynema_sgf::actuator::utils::ActParser pp(
            "Actuator.ActuatorSector", "Actuator.R1");
        sector.read_inputs(pp);
    }

    amrex::Vector<int> act_proc_count(amrex::ParallelDescriptor::NProcs(), 0);
    sector.determine_root_proc(act_proc_count);
    sector.init_actuator_source();

    const auto& meta = sector.meta();
    EXPECT_GT(meta.radius.size(), 0);
    EXPECT_EQ(meta.radius.size(), meta.dr.size());
    EXPECT_EQ(meta.gaussian_table_nintervals, 319);
    EXPECT_EQ(meta.gaussian_table.size(), meta.gaussian_table_nintervals + 1);
    EXPECT_EQ(sector.num_velocity_points(), 2 * meta.radius.size());

    const auto refinement = sector.refinement_geometries(0.5_rt);
    ASSERT_EQ(refinement.size(), 1U);
    EXPECT_EQ(refinement[0].label, "R1");
    constexpr amrex::Real tol = kynema_sgf::constants::TIGHT_TOL;
    EXPECT_NEAR(refinement[0].center.x(), 0.0_rt, tol);
    EXPECT_NEAR(refinement[0].center.y(), 0.0_rt, tol);
    EXPECT_NEAR(refinement[0].center.z(), 0.5_rt, tol);
    EXPECT_NEAR(refinement[0].normal.x(), 1.0_rt, tol);
    EXPECT_NEAR(refinement[0].normal.y(), 0.0_rt, tol);
    EXPECT_NEAR(refinement[0].normal.z(), 0.0_rt, tol);
    EXPECT_NEAR(refinement[0].rotor_radius, 0.05_rt, tol);
    EXPECT_NEAR(refinement[0].epsilon_max, 0.0025_rt, tol);

    remove_file(m_afname);
}

TEST_F(ActuatorSectorTest, actuator_lifecycle)
{
    write_airfoil_file(m_afname);
    initialize_domain();
    populate_actuator_inputs();

    ActPhysicsTest act(sim());
    act.pre_init_actions();
    act.post_init_actions();
    act.pre_advance_work();

    remove_file(m_afname);
}

TEST_F(ActuatorSectorRefinementTest, asymmetric_actuator_refinement)
{
    write_airfoil_file(m_afname);
    initialize_domain();
    populate_actuator_inputs();

    auto& physics = sim().physics_manager().create("Actuator", sim());
    auto& actuator = dynamic_cast<::kynema_sgf::actuator::Actuator&>(physics);
    actuator.pre_init_actions();

    amrex::ParmParse pp("tagging.rotor_tracking");
    pp.addarr("actuator_labels", amrex::Vector<std::string>{"R1"});
    pp.add("min_level", 0);
    pp.add("max_level", 0);
    pp.add("radial_padding_epsilon", 0.0_rt);
    pp.add("forward_padding_diameter", 0.8_rt);
    pp.add("backward_padding_diameter", 0.3_rt);

    auto refiner =
        ::kynema_sgf::RefinementCriteria::create("ActuatorRefinement", sim());
    refiner->initialize("tagging.rotor_tracking");

    const auto& field = sim().repo().get_field("velocity")(0);
    amrex::TagBoxArray tags(field.boxArray(), field.DistributionMap());
    tags.setVal(amrex::TagBox::CLEAR);
    (*refiner)(0, tags, 0.0_rt, 0);
    amrex::Gpu::streamSynchronize();

    amrex::Gpu::PinnedVector<amrex::IntVect> tagged_cells;
    tags.collate(tagged_cells);
    ASSERT_EQ(tagged_cells.size(), 48U);

    const auto& geom = sim().mesh().Geom(0);
    const auto& problo = geom.ProbLoArray();
    const auto& dx = geom.CellSizeArray();
    amrex::Real xmin = std::numeric_limits<amrex::Real>::max();
    amrex::Real xmax = std::numeric_limits<amrex::Real>::lowest();
    for (const auto& cell : tagged_cells) {
        const amrex::Real x = problo[0] + (cell[0] + 0.5_rt) * dx[0];
        xmin = amrex::min(xmin, x);
        xmax = amrex::max(xmax, x);
    }
    constexpr amrex::Real tol = kynema_sgf::constants::TIGHT_TOL;
    EXPECT_NEAR(xmin, -0.0125_rt, tol);
    EXPECT_NEAR(xmax, 0.0625_rt, tol);

    remove_file(m_afname);
}

TEST_F(ActuatorSectorTest, fixed_epsilon_input)
{
    write_airfoil_file(m_afname);
    initialize_domain();
    populate_actuator_inputs(true, 0.004_rt, false);

    ::kynema_sgf::actuator::ActModel<
        ::kynema_sgf::actuator::ActuatorSector,
        ::kynema_sgf::actuator::ActSrcSector>
        sector(sim(), "R1", 0);
    {
        ::kynema_sgf::actuator::utils::ActParser pp(
            "Actuator.ActuatorSector", "Actuator.R1");
        sector.read_inputs(pp);
    }
    sector.init_actuator_source();

    const auto& meta = sector.meta();
    for (const auto eps : meta.epsilon_profile) {
        for (int n = 0; n < 3; ++n) EXPECT_NEAR(eps[n], 0.004_rt, 1.0e-14_rt);
    }

    remove_file(m_afname);
}

TEST_F(ActuatorSectorTest, epsilon_chord_uses_minimum)
{
    write_airfoil_file(m_afname);
    initialize_domain();
    populate_actuator_inputs(false, 0.0_rt, true, 0.25_rt, 0.004_rt);

    ::kynema_sgf::actuator::ActModel<
        ::kynema_sgf::actuator::ActuatorSector,
        ::kynema_sgf::actuator::ActSrcSector>
        sector(sim(), "R1", 0);
    {
        ::kynema_sgf::actuator::utils::ActParser pp(
            "Actuator.ActuatorSector", "Actuator.R1");
        sector.read_inputs(pp);
    }
    sector.init_actuator_source();

    const auto& meta = sector.meta();
    for (int i = 0; i < static_cast<int>(meta.epsilon_profile.size()); ++i) {
        const amrex::Real expected =
            amrex::max(0.004_rt, 0.25_rt * meta.chord[i]);
        for (int n = 0; n < 3; ++n)
            EXPECT_NEAR(meta.epsilon_profile[i][n], expected, 1.0e-14_rt);
    }

    remove_file(m_afname);
}

TEST_F(ActuatorSectorTest, directional_epsilon_follows_blade)
{
    namespace act = kynema_sgf::actuator;
    using kynema_sgf::vs::Vector;
    write_airfoil_file(m_afname);
    initialize_domain();
    populate_actuator_inputs();
    amrex::ParmParse pp("Actuator.R1");
    pp.addarr("epsilon", act::RealList{0.025, 0.035, 0.045});
    pp.addarr("epsilon_chord", act::RealList{4, 0, 0});
    pp.addarr("rotor_normal", act::RealList{1, 2, 3});
    pp.addarr("translation_velocity", act::RealList{0, 0, 0});
    pp.add("support_radius_over_epsilon", 5.0);
    auto& field = sim().repo().get_field("actuator_src_term");
    auto& moments = sim().repo().declare_field("kernel_moments", 3, 0);
    const auto& geom = sim().mesh().Geom(0);
    const auto dx = geom.CellSizeArray();
    const auto lo = geom.ProbLoArray();
    const auto dv = dx[0] * dx[1] * dx[2];
    for (int sign : {-1, 1}) {
        pp.add("omega", sign * 2500.0);
        pp.addarr("twist", act::RealList{sign * 30.0, sign * 30.0});
        pp.add("gaussian_type", std::string(sign < 0 ? "direct" : "table"));
        act::ActuatorSector::DataType data(sim(), "R1", 0);
        act::ops::ReadInputsOp<act::ActuatorSector, act::ActSrcSector>()(
            data,
            act::utils::ActParser("Actuator.ActuatorSector", "Actuator.R1"));
        act::ops::InitDataOp<act::ActuatorSector, act::ActSrcSector>()(data);
        act::ops::ComputeForceOp<act::ActuatorSector, act::ActSrcSector>()(
            data);
        auto& grid = data.grid();
        const auto& meta = data.meta();
        for (int i = 0; i < static_cast<int>(grid.pos.size()); ++i) {
            const auto radial = (grid.pos[i] - meta.center).unit();
            const auto normal = meta.rotor_normal.unit();
            const auto tangent = radial ^ normal;
            const auto chord = std::sqrt(0.75) * tangent - sign * 0.5 * normal;
            const auto thickness =
                sign * 0.5 * tangent + std::sqrt(0.75) * normal;
            for (int n = 0; n < 3; ++n) {
                EXPECT_NEAR(
                    (grid.orientation[i] & chord)[n], n == 0 ? 1 : 0, 1e-12);
                EXPECT_NEAR(
                    (grid.orientation[i] & thickness)[n], n == 1 ? 1 : 0,
                    1e-12);
                EXPECT_NEAR(
                    (grid.orientation[i] & radial)[n], n == 2 ? 1 : 0, 1e-12);
            }
        }
        const auto eps = grid.epsilon.front();
        EXPECT_NEAR(eps.x(), std::max(0.025, 4 * meta.chord.front()), 1e-12);
        EXPECT_NEAR(eps.y(), 0.035, 1e-12);
        EXPECT_NEAR(eps.z(), 0.045, 1e-12);
        // Isolate one rotated kernel: verify its integral and directional
        // variance.
        const auto orientation = grid.orientation.front();
        grid.pos.assign(1, Vector::zero());
        grid.force.assign(1, Vector{0, 0, 1});
        grid.epsilon.resize(1);
        grid.orientation.resize(1);
        act::ops::ActSrcOp<act::ActuatorSector, act::ActSrcSector> source(data);
        source.initialize();
        field.setVal(0);
        for (amrex::MFIter mfi(field(0)); mfi.isValid(); ++mfi) {
            source(0, mfi, geom);
            const auto f = field(0).const_array(mfi);
            const auto moment = moments(0).array(mfi);
            amrex::ParallelFor(
                mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    const auto local =
                        orientation & Vector{
                                          lo[0] + (i + 0.5) * dx[0],
                                          lo[1] + (j + 0.5) * dx[1],
                                          lo[2] + (k + 0.5) * dx[2]};
                    for (int n = 0; n < 3; ++n)
                        moment(i, j, k, n) =
                            f(i, j, k, 2) * local[n] * local[n];
                });
        }
        EXPECT_NEAR(field(0).sum(2) * dv, 1, 2e-4);
        for (int n = 0; n < 3; ++n)
            EXPECT_NEAR(
                moments(0).sum(n) * dv, 0.5 * eps[n] * eps[n],
                2e-4 * eps[n] * eps[n]);
        // Radial smoothing must not change the swept-point count.
        const auto counts = meta.theta_counts;
        for (auto& width : data.meta().epsilon_profile) width.z() = 1e-4;
        act::ops::ComputeForceOp<act::ActuatorSector, act::ActSrcSector>()(
            data);
        EXPECT_EQ(meta.theta_counts, counts);
    }
    remove_file(m_afname);
}

} // namespace kynema_sgf_tests
