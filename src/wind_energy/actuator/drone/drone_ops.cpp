#include "src/wind_energy/actuator/drone/drone_ops.H"

#include "src/CFDSim.H"
#include "src/utilities/constants.H"
#include "src/utilities/io_utils.H"
#include "src/utilities/ncutils/nc_interface.H"

#include "src/wind_energy/actuator/wing/fixed_wing_ops.H"
#include "src/utilities/trig_ops.H"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "AMReX_Print.H"

using namespace amrex::literals;

namespace kynema_sgf::actuator::ops {
namespace {

// One physical sample cloud; the aerodynamic owner remains fixed on regrid.
void select_sampling_owner(Drone::DataType& data)
{
    auto& info = data.info();
    const auto mesh_procs = utils::determine_influenced_procs(
        data.sim().mesh(), data.sim().mesh().Geom(0).ProbDomain());
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        !mesh_procs.empty(), "Drone sampling requires a nonempty mesh");
    auto owner = info.root_proc;
    if (!mesh_procs.contains(owner)) {
        const auto candidate =
            std::ranges::find_if(info.procs, [&](const int rank) {
                return mesh_procs.contains(rank);
            });
        owner =
            candidate != info.procs.end() ? *candidate : *mesh_procs.begin();
    }
    data.meta().sampling_proc = owner;
    info.procs.insert(owner);
    const auto rank = amrex::ParallelDescriptor::MyProc();
    info.actuator_in_proc = info.procs.contains(rank);
    info.sample_vel_in_proc = rank == owner;
}

void share_sampled_fields(Drone::DataType& data)
{
    auto& meta = data.meta();
    const auto& info = data.info();
    const auto rank = amrex::ParallelDescriptor::MyProc();
    if (meta.sampling_proc < 0 ||
        (rank != meta.sampling_proc && !info.is_root_proc)) {
        return;
    }
    BL_PROFILE("kynema-sgf::actuator::Drone::share_sampled_fields");
    // The root may delegate sampling when it has no mesh cells.
    UpdatePosOp<Drone, ActSrcDrone>()(data);
#ifdef AMREX_USE_MPI
    if (meta.sampling_proc != info.root_proc) {
        auto& grid = data.grid();
        auto& buffer = meta.sampling_buffer;
        const auto count = grid.vel.size();
        buffer.resize(4 * count);
        AMREX_ALWAYS_ASSERT(meta.sampling_comm);
        constexpr int sample_tag = 1001;
        if (rank == meta.sampling_proc) {
            for (amrex::Long i = 0; i < count; ++i) {
                for (int d = 0; d < 3; ++d) {
                    buffer[(4 * i) + d] = grid.vel[i][d];
                }
                buffer[(4 * i) + 3] = grid.density[i];
            }
            amrex::ParallelDescriptor::Send(
                buffer.data(), buffer.size(), info.root_proc, sample_tag,
                *meta.sampling_comm);
        } else {
            amrex::ParallelDescriptor::Recv(
                buffer.data(), buffer.size(), meta.sampling_proc, sample_tag,
                *meta.sampling_comm);
            for (amrex::Long i = 0; i < count; ++i) {
                for (int d = 0; d < 3; ++d) {
                    grid.vel[i][d] = buffer[(4 * i) + d];
                }
                grid.density[i] = buffer[(4 * i) + 3];
            }
        }
    }
#endif
    if (info.is_root_proc) {
        UpdateVelOp<Drone, ActSrcDrone>()(data);
    }
}

// Bound the actual quadrature points, including motion throughout the step.
// A sphere using the largest width contains every rotated anisotropic kernel.
amrex::RealBox projection_bounds(
    const ActGrid& grid, const amrex::RealBox& domain, const amrex::Real cutoff)
{
    if (grid.pos.empty()) {
        return domain;
    }
    vs::Vector lower = grid.pos.front(), upper = lower;
    for (int i = 0; i < static_cast<int>(grid.pos.size()); ++i) {
        const auto& frame = grid.orientation[i];
        const std::array rows{frame.x(), frame.y(), frame.z()};
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                if (!(std::abs(
                          (rows[j] & rows[k]) - (j == k ? 1.0_rt : 0.0_rt)) <
                      64 * std::numeric_limits<amrex::Real>::epsilon())) {
                    return domain;
                }
            }
        }
        const auto& eps = grid.epsilon[i];
        const auto extent = cutoff * std::max({eps.x(), eps.y(), eps.z()});
        if (!std::isfinite(extent)) {
            return domain;
        }
        for (int d = 0; d < 3; ++d) {
            lower[d] = std::min(lower[d], grid.pos[i][d] - extent);
            upper[d] = std::max(upper[d], grid.pos[i][d] + extent);
        }
    }
    return {lower.data(), upper.data()};
}

amrex::RealBox structure_bounds(const Drone::DataType& data)
{
    const auto& domain = data.sim().mesh().Geom(0).ProbDomain();
    for (const auto& arm : data.meta().arms) {
        for (const auto flag : arm->grid().dcoord_flags) {
            if (flag != 1.0_rt) {
                return domain;
            }
        }
    }
    return projection_bounds(data.grid(), domain, 4.0_rt);
}

void distribute_projection(Drone::DataType& data)
{
    auto& info = data.info();
    // Unregistered operators retain their local evaluation path.
    if (info.root_proc < 0) {
        return;
    }
    BL_PROFILE("kynema-sgf::actuator::Drone::distribute_projection");
    auto& meta = data.meta();
    amrex::Vector<ActGrid*> grids{&data.grid()};
    for (auto& rotor : meta.rotors) {
        grids.push_back(&rotor->data.grid());
    }
    amrex::Vector<int> counts(grids.size()), procs;
    RealList bounds(6 * grids.size());
    int num_procs = 0;
    if (info.is_root_proc) {
        const auto& domain = data.sim().mesh().Geom(0).ProbDomain();
        vs::Vector lo{domain.hi(0), domain.hi(1), domain.hi(2)};
        vs::Vector hi{domain.lo(0), domain.lo(1), domain.lo(2)};
        for (int g = 0; g < static_cast<int>(grids.size()); ++g) {
            counts[g] = static_cast<int>(grids[g]->pos.size());
            const auto box = g == 0 ? structure_bounds(data)
                                    : projection_bounds(
                                          *grids[g], domain,
                                          meta.rotors[g - 1]
                                              ->data.meta()
                                              .support_radius_over_epsilon);
            for (int d = 0; d < 3; ++d) {
                bounds[(6 * g) + d] = box.lo(d);
                bounds[(6 * g) + 3 + d] = box.hi(d);
                if (counts[g] > 0) {
                    lo[d] = std::min(lo[d], box.lo(d));
                    hi[d] = std::max(hi[d], box.hi(d));
                }
            }
        }
        info.bound_box = amrex::RealBox(lo.data(), hi.data());
        info.procs = utils::determine_influenced_procs(
            data.sim().mesh(), info.bound_box);
        info.procs.insert(info.root_proc);
        info.procs.insert(meta.sampling_proc);
        procs.assign(info.procs.begin(), info.procs.end());
        num_procs = static_cast<int>(procs.size());
        if (data.sim().time().time_index() == 0) {
            amrex::Print(info.root_proc)
                << "Drone " << info.label << ": aerodynamic owner "
                << info.root_proc << ", sampling owner " << meta.sampling_proc
                << ", projection ranks " << num_procs << "/"
                << amrex::ParallelDescriptor::NProcs() << '\n';
        }
    }
#ifdef AMREX_USE_MPI
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        const MPI_Comm comm = *meta.sampling_comm;
        amrex::ParallelDescriptor::Bcast(
            counts.data(), counts.size(), info.root_proc, comm);
        amrex::ParallelDescriptor::Bcast(
            bounds.data(), bounds.size(), info.root_proc, comm);
        amrex::ParallelDescriptor::Bcast(&num_procs, 1, info.root_proc, comm);
        procs.resize(num_procs);
        amrex::ParallelDescriptor::Bcast(
            procs.data(), procs.size(), info.root_proc, comm);
    }
#endif
    info.procs = std::set<int>(procs.begin(), procs.end());
    info.actuator_in_proc =
        info.procs.contains(amrex::ParallelDescriptor::MyProc());
    // Keep current bounds on all ranks for the existing regrid callback.
    const auto& domain = data.sim().mesh().Geom(0).ProbDomain();
    vs::Vector lo{domain.hi(0), domain.hi(1), domain.hi(2)};
    vs::Vector hi{domain.lo(0), domain.lo(1), domain.lo(2)};
    for (int g = 0; g < static_cast<int>(grids.size()); ++g) {
        const auto offset = 6 * static_cast<size_t>(g);
        const amrex::RealBox box(
            bounds.data() + offset, bounds.data() + offset + 3);
        if (g > 0) {
            meta.rotors[g - 1]->data.info().bound_box = box;
        }
        if (counts[g] == 0) {
            continue;
        }
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], box.lo(d));
            hi[d] = std::max(hi[d], box.hi(d));
        }
    }
    info.bound_box = amrex::RealBox(lo.data(), hi.data());
    if (!info.actuator_in_proc) {
        return;
    }
    auto& buffer = meta.projection_buffer;
    amrex::Long total = 0;
    for (const int n : counts) {
        total += n;
    }
    AMREX_ALWAYS_ASSERT(total <= std::numeric_limits<int>::max() / 18);
    buffer.resize(18 * total);
    const auto transfer = [&]() {
        size_t offset = 0;
        const auto value = [&](amrex::Real& v) {
            if (info.is_root_proc) {
                buffer[offset] = v;
            } else {
                v = buffer[offset];
            }
            ++offset;
        };
        for (int g = 0; g < static_cast<int>(grids.size()); ++g) {
            auto& grid = *grids[g];
            grid.pos.resize(counts[g]);
            grid.force.resize(counts[g]);
            grid.epsilon.resize(counts[g]);
            grid.orientation.resize(counts[g]);
            for (int i = 0; i < counts[g]; ++i) {
                for (auto* vector :
                     {&grid.pos[i], &grid.force[i], &grid.epsilon[i]}) {
                    for (auto& v : *vector) {
                        value(v);
                    }
                }
                for (auto& v : grid.orientation[i]) {
                    value(v);
                }
            }
        }
    };
    if (info.is_root_proc) {
        transfer();
    }
#ifdef AMREX_USE_MPI
    constexpr int projection_tag = 1002;
    if (amrex::ParallelDescriptor::NProcs() > 1) {
        if (info.is_root_proc) {
            for (const int rank : procs) {
                if (rank != info.root_proc) {
                    amrex::ParallelDescriptor::Send(
                        buffer.data(), buffer.size(), rank, projection_tag,
                        *meta.sampling_comm);
                }
            }
        } else {
            amrex::ParallelDescriptor::Recv(
                buffer.data(), buffer.size(), info.root_proc, projection_tag,
                *meta.sampling_comm);
        }
    }
#endif
    if (!info.is_root_proc) {
        transfer();
    }
}

void input_error(const std::string& label, const std::string& message)
{
    amrex::Abort("Drone '" + label + "': " + message);
}

void validate_distinct_angles(const std::string& label, const RealList& angles)
{
    for (int i = 0; i < static_cast<int>(angles.size()); ++i) {
        for (int j = i + 1; j < static_cast<int>(angles.size()); ++j) {
            amrex::Real delta = std::remainder(angles[i] - angles[j], 360.0_rt);
            if (std::abs(delta) < constants::TIGHT_TOL) {
                input_error(label, "arm angles must be distinct");
            }
        }
    }
}

// Store geometry in the drone frame; evaluate wings in their local axes.
void read_structure(Drone::DataType& data, const utils::ActParser& pp)
{
    auto& m = data.meta();
    if (!pp.contains("body_dimensions")) {
        return;
    }
    RealList size;
    amrex::Vector<int> count;
    vs::Vector center, angles;
    amrex::Real cd, minimum = 0, limit = 0.38_rt;
    std::string epsilon;
    pp.getarr("body_dimensions", size);
    pp.getarr("body_num_points", count);
    pp.query("body_offset", center);
    pp.query("body_orientation", angles);
    pp.get("body_cd", cd);
    pp.get("body_epsilon", epsilon);
    pp.query("body_epsilon_min", minimum);
    pp.query("body_nonlinear_limit", limit);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        size.size() == 2 && count.size() == 2,
        "body_dimensions and body_num_points each require two entries");
    for (auto v : size) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(v) && v > 0,
            "body_dimensions must be finite and positive");
    }
    for (int d = 0; d < 3; ++d) {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(center[d]) && std::isfinite(angles[d]),
            "body_offset and body_orientation must be finite");
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        count[0] > 0 && count[1] > 0, "body_num_points must be positive");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(cd) && cd >= 0 && std::isfinite(minimum) &&
            minimum >= 0 && std::isfinite(limit) && limit > 0,
        "body_cd and body_epsilon_min must be finite and nonnegative; "
        "body_nonlinear_limit must be finite and positive");
    const auto rotation = drone::body_rotation(angles);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        count[0] <= std::numeric_limits<int>::max() / count[1],
        "body_num_points product exceeds the supported integer range");
    m.box_points = count[0] * count[1];
    m.box_drag_area = cd * size[0] * size[1] / m.box_points;
    if (epsilon == "momentum_thickness") {
        m.box_epsilon = std::sqrt(2 * m.box_drag_area / std::acos(-1.0_rt));
    } else {
        pp.get("body_epsilon", m.box_epsilon);
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m.box_epsilon) && m.box_epsilon >= 0,
        "body_epsilon must be finite and nonnegative");
    m.box_epsilon = std::max(
        {m.box_epsilon, minimum, std::sqrt(m.box_drag_area / (2 * limit))});
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        std::isfinite(m.box_epsilon) && m.box_epsilon > 0,
        "Final body epsilon must be finite and positive; set body_epsilon_min "
        "for zero-drag momentum_thickness mode");
    for (int i = 0; i < count[0]; ++i) {
        for (int j = 0; j < count[1]; ++j) {
            m.body_points.push_back(
                center +
                (rotation & vs::Vector{
                                ((i + 0.5_rt) / count[0] - 0.5_rt) * size[0],
                                ((j + 0.5_rt) / count[1] - 0.5_rt) * size[1],
                                0}));
        }
    }
    utils::ActParser arm(
        "Actuator.Drone.arm", "Actuator." + data.info().label + ".arm");
    if (!arm.contains("airfoil_table")) {
        return;
    }
    // Supply generated geometry to the unchanged wing reader.
    const auto prefix = "Actuator." + data.info().label + ".arm";
    const auto section_prefix = prefix + "_section";
    amrex::ParmParse section(section_prefix);
    for (const auto& name : amrex::ParmParse::getEntries(prefix)) {
        amrex::Vector<std::string> values;
        const auto key = name.substr(prefix.size() + 1);
        arm.params().getarr(key, values);
        section.addarr(key, values);
    }
    const utils::ActParser wing_pp("Actuator.Drone.arm", section_prefix);
    for (const auto& rotor : m.rotors) {
        auto wing = std::make_unique<FixedWing::DataType>(
            data.sim(), data.info().label + ".arm", data.info().id);
        auto& w = wing->meta();
        auto start = vs::Vector::zero();
        auto end = rotor->body_offset;
        const auto phase = vs::zrot(m.arm_angles_degrees[m.arms.size()]);
        if (arm.contains("end")) {
            arm.get("end", end);
            end = phase & end;
        }
        if (arm.contains("start")) {
            arm.get("start", start);
            start = phase & start;
        } else {
            const auto c = center & rotation;
            const auto h = end & rotation;
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(c.z()) < constants::TIGHT_TOL &&
                    std::abs(h.z()) < constants::TIGHT_TOL,
                "Automatic arms require coplanar box and hub rays; supply "
                "arm.start otherwise");
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                std::abs(c.x()) < size[0] / 2 && std::abs(c.y()) < size[1] / 2,
                "Automatic arms require the drone origin inside the box; "
                "supply arm.start otherwise");
            amrex::Real fraction = 1;
            for (int d = 0; d < 2; ++d) {
                const auto edge = c[d] + std::copysign(size[d] / 2, h[d]);
                if (h[d] != 0) fraction = std::min(fraction, edge / h[d]);
            }
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                fraction > 0 && fraction < 1,
                "Arm end must lie outside the box");
            start = fraction * end;
        }
        const auto length = vs::mag(end - start);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            std::isfinite(length) && length > 0,
            "Arm endpoints must define a finite, nonzero span");
        const auto span = (end - start) / length;
        const auto chord = (span ^ vs::Vector::khat()).unit();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            vs::mag(chord) > 0, "Vertical reference arms are not supported");
        const vs::Tensor frame(chord, span, chord ^ span, true);
        section.addarr("start", RealList{0, 0, 0});
        section.addarr("end", RealList{0, length, 0});
        ReadInputsOp<FixedWing, ActSrcLine>()(*wing, wing_pp);
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            w.num_pts >= 2, "arm.num_points must be at least two");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            amrex::toLower(w.motion_type) == "none",
            "Arms follow the drone motion; arm.motion_type must be none");
        InitDataOp<FixedWing, ActSrcLine>()(*wing);
        for (int i = 0; i < w.num_pts; ++i) {
            for (auto e : wing->grid().epsilon[i]) {
                AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                    std::isfinite(e) && e > 0,
                    "arm.epsilon or arm.epsilon_chord must give positive "
                    "finite widths in all directions");
            }
            m.body_points.push_back(start + (frame & wing->grid().pos[i]));
        }
        m.arm_frames.push_back(frame);
        m.arms.push_back(std::move(wing));
    }
}

void prepare_netcdf_file(
    const std::string& filename, const Drone::DataType& data)
{
#ifdef KYNEMA_SGF_USE_NETCDF
    auto ncf = ncutils::NCFile::create(filename, NC_CLOBBER | NC_NETCDF4);
    const std::string nt_name{"num_time_steps"};

    ncf.enter_def_mode();
    ncf.put_attr("title", "Kynema-SGF Drone actuator-load output");
    ncf.put_attr("version", ioutils::kynema_sgf_version());
    ncf.put_attr("created_on", ioutils::timestamp());
    ncf.def_dim(nt_name, NC_UNLIMITED);
    ncf.def_dim("ndim", AMREX_SPACEDIM);

    auto grp = ncf.def_group(data.info().label);
    grp.def_var("time", NC_DOUBLE, {nt_name});
    grp.def_var("time_index", NC_INT, {nt_name});
    grp.def_var("force", NC_DOUBLE, {nt_name, "ndim"});
    grp.def_var("moment", NC_DOUBLE, {nt_name, "ndim"});
    grp.var("force").put_attr(
        "description",
        "total density-normalized aerodynamic force acting on the vehicle");
    grp.var("force").put_attr("units", "m^4 s^-2");
    grp.var("moment").put_attr(
        "description",
        "total density-normalized aerodynamic moment about the drone center");
    grp.var("moment").put_attr("units", "m^5 s^-2");
    for (int i = 0; i < static_cast<int>(data.meta().rotors.size()); ++i) {
        sector::prepare_netcdf_group(
            grp, "R" + std::to_string(i + 1),
            data.meta().rotors[i]->data.meta());
    }
    ncf.exit_def_mode();
    for (int i = 0; i < static_cast<int>(data.meta().rotors.size()); ++i) {
        sector::write_netcdf_group_metadata(
            grp, "R" + std::to_string(i + 1),
            data.meta().rotors[i]->data.meta());
    }
    ncf.close();
#else
    amrex::ignore_unused(filename, data);
#endif
}

void write_netcdf(
    const std::string& filename,
    const Drone::DataType& data,
    const amrex::Real time,
    const int time_index)
{
#ifdef KYNEMA_SGF_USE_NETCDF
    auto ncf = ncutils::NCFile::open(filename, NC_WRITE);
    const size_t nt = ncf.dim("num_time_steps").len();
    auto grp = ncf.group(data.info().label);
    const auto& force = data.meta().total_force;
    const auto& moment = data.meta().total_moment;

    grp.var("time").put(&time, {nt}, {1});
    grp.var("time_index").put(&time_index, {nt}, {1});
    grp.var("force").put(force.data(), {nt, 0}, {1, AMREX_SPACEDIM});
    grp.var("moment").put(moment.data(), {nt, 0}, {1, AMREX_SPACEDIM});
    for (int i = 0; i < static_cast<int>(data.meta().rotors.size()); ++i) {
        const auto& rotor = data.meta().rotors[i]->data;
        sector::write_netcdf_group(
            grp, "R" + std::to_string(i + 1), rotor.meta(), rotor.grid(), time,
            time_index, nt);
    }
    ncf.close();
#else
    amrex::ignore_unused(filename, data, time, time_index);
#endif
}

} // namespace

amrex::Vector<ActuatorRefinementGeometry>
RefinementGeometryOp<Drone, ActSrcDrone>::operator()(
    const Drone::DataType& data, const amrex::Real time) const
{
    amrex::Vector<ActuatorRefinementGeometry> geometries;
    geometries.reserve(data.meta().rotors.size());
    for (const auto& rotor : data.meta().rotors) {
        geometries.emplace_back(
            sector::refinement_geometry(
                rotor->data.meta(), rotor->data.info().label, time));
    }
    return geometries;
}

std::optional<motion::RigidTransform>
ReferenceFrameOp<Drone, ActSrcDrone>::operator()(
    const Drone::DataType& data, const amrex::Real time) const
{
    const auto& motion = data.meta().body_motion;
    if (motion == nullptr) {
        return std::nullopt;
    }
    return motion->pose(time);
}

void ReadInputsOp<Drone, ActSrcDrone>::operator()(
    Drone::DataType& data, const utils::ActParser& pp)
{
    auto& meta = data.meta();
    data.info().force_update_all_ranks = true;
    const auto& label = data.info().label;

    pp.get("num_rotors", meta.num_rotors);
    if (meta.num_rotors <= 0) {
        input_error(label, "num_rotors must be positive");
    }
    if (!pp.contains("position_timetable")) {
        pp.get("center", meta.center);
    }
    pp.query("translation_velocity", meta.translation_velocity);
    pp.query("body_orientation_degrees", meta.body_orientation_degrees);
    if (pp.contains("orientation_timetable") &&
        pp.contains("body_orientation_degrees")) {
        input_error(
            label,
            "body_orientation_degrees cannot be combined with "
            "orientation_timetable");
    }
    meta.body_orientation = drone::body_rotation(meta.body_orientation_degrees);
    // All child sectors share these histories, keeping the rigid-body pose and
    // rotor clocks synchronized across the composite actuator.
    meta.body_motion = std::make_shared<motion::RigidBodyMotion>();
    meta.body_motion->read_inputs(
        pp, meta.center, meta.body_orientation, "angular_velocity");
    meta.rotor_motion = std::make_shared<motion::RotorMotion>();
    meta.rotor_motion->read_drone_inputs(pp, meta.num_rotors);

    // One arm length applies to every rotor; num_rotors values permit an
    // asymmetric layout. MultiParser preserves instance-over-default priority.
    if (!pp.contains("arm_length")) {
        input_error(label, "arm_length is required");
    }
    pp.getarr("arm_length", meta.arm_lengths);
    if (meta.arm_lengths.size() == 1) {
        const auto arm_length = meta.arm_lengths.front();
        meta.arm_lengths.assign(meta.num_rotors, arm_length);
    }
    if (static_cast<int>(meta.arm_lengths.size()) != meta.num_rotors) {
        input_error(
            label,
            "arm_length must contain either one value or num_rotors values");
    }
    if (std::ranges::any_of(meta.arm_lengths, [](const amrex::Real length) {
            return length <= 0.0_rt;
        })) {
        input_error(label, "all arm lengths must be positive");
    }

    // Explicit angles define the base arm pattern. If they are omitted, use a
    // uniform pattern. Phase then rotates the complete pattern in the body
    // x-y plane without changing its relative geometry.
    if (pp.contains("arm_angles_degrees")) {
        pp.getarr("arm_angles_degrees", meta.arm_angles_degrees);
    } else {
        meta.arm_angles_degrees =
            drone::uniform_arm_angles(meta.num_rotors, 0.0_rt);
    }
    if (static_cast<int>(meta.arm_angles_degrees.size()) != meta.num_rotors) {
        input_error(label, "arm_angles_degrees must contain num_rotors values");
    }
    pp.query("arm_phase_degrees", meta.arm_phase_degrees);
    for (auto& angle : meta.arm_angles_degrees) {
        angle += meta.arm_phase_degrees;
    }
    validate_distinct_angles(label, meta.arm_angles_degrees);

    meta.mirror_blades.assign(meta.num_rotors, 0);
    if (pp.contains("mirror_blades")) {
        amrex::Vector<std::string> mirror_inputs;
        pp.getarr("mirror_blades", mirror_inputs);
        if (static_cast<int>(mirror_inputs.size()) != meta.num_rotors) {
            input_error(label, "mirror_blades must contain num_rotors values");
        }
        for (int i = 0; i < meta.num_rotors; ++i) {
            const auto value = amrex::toLower(mirror_inputs[i]);
            if ((value != "true") && (value != "false")) {
                input_error(
                    label, "mirror_blades values must be true or false");
            }
            meta.mirror_blades[i] = (value == "true") ? 1 : 0;
        }
    }

    const auto offsets =
        drone::rotor_body_offsets(meta.arm_lengths, meta.arm_angles_degrees);
    meta.rotors.reserve(meta.num_rotors);
    for (int i = 0; i < meta.num_rotors; ++i) {
        const std::string rotor_label = label + ".R" + std::to_string(i + 1);
        auto rotor = std::make_unique<DroneRotor>(
            data.sim(), rotor_label, (data.info().id * 100000) + i);
        rotor->body_offset = offsets[i];

        // A drone owns its rotor configuration. Reuse the same default and
        // instance namespaces for both drone geometry and sector aerodynamics;
        // each reader simply ignores inputs outside its responsibility.
        utils::ActParser sector_pp("Actuator.Drone", "Actuator." + label);
        auto& rotor_meta = rotor->data.meta();
        rotor_meta.body_motion = meta.body_motion;
        rotor_meta.rotor_motion = meta.rotor_motion;
        rotor_meta.rotor_index = i;
        rotor_meta.body_offset = offsets[i];
        rotor_meta.omega = meta.rotor_motion->omega(i, 0.0_rt);
        rotor_meta.user_omega = true;
        ReadInputsOp<ActuatorSector, ActSrcSector>()(rotor->data, sector_pp);
        if (meta.mirror_blades[i] != 0) {
            // Mirroring the blade geometry reverses twist without introducing
            // a separate rotation-direction convention.
            for (auto& twist : rotor->data.meta().twist_inp) {
                twist = -twist;
            }
        }
        const auto center =
            meta.body_motion->position(0.0_rt) +
            (meta.body_motion->orientation(0.0_rt) & rotor->body_offset);
        const auto rotor_normal =
            meta.body_motion->orientation(0.0_rt) & vs::Vector::khat();
        sector::set_placement(
            rotor->data, center, rotor_normal,
            meta.body_motion->translation_velocity(0.0_rt));
        rotor->output.read_io_options(sector_pp);
        meta.rotors.emplace_back(std::move(rotor));
    }

    read_structure(data, pp);

    // The composite search region is the union of its child-sector regions.
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> lo;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> hi;
    for (int n = 0; n < AMREX_SPACEDIM; ++n) {
        lo[n] = std::numeric_limits<amrex::Real>::max();
        hi[n] = std::numeric_limits<amrex::Real>::lowest();
    }
    for (const auto& rotor : meta.rotors) {
        const auto& box = rotor->data.info().bound_box;
        for (int n = 0; n < AMREX_SPACEDIM; ++n) {
            lo[n] = std::min(lo[n], box.lo(n));
            hi[n] = std::max(hi[n], box.hi(n));
        }
    }
    data.info().bound_box = amrex::RealBox(lo.data(), hi.data());
    // Moving structural kernels may extend beyond rotor search regions.
    if (!meta.body_points.empty())
        data.info().bound_box = data.sim().mesh().Geom(0).ProbDomain();
}

template <>
void determine_root_proc<Drone>(
    Drone::DataType& data, amrex::Vector<int>& act_proc_count)
{
    auto& info = data.info();
    info.procs =
        utils::determine_influenced_procs(data.sim().mesh(), info.bound_box);
    utils::determine_root_proc(info, act_proc_count);
    select_sampling_owner(data);
}

template <>
void determine_influenced_procs<Drone>(Drone::DataType& data)
{
    auto& info = data.info();
    info.procs =
        utils::determine_influenced_procs(data.sim().mesh(), info.bound_box);
    if (info.root_proc >= 0) {
        info.procs.insert(info.root_proc);
    }
    select_sampling_owner(data);
}

void InitDataOp<Drone, ActSrcDrone>::operator()(Drone::DataType& data)
{
    auto& meta = data.meta();
#ifdef AMREX_USE_MPI
    if (data.info().root_proc >= 0 && amrex::ParallelDescriptor::NProcs() > 1 &&
        !meta.sampling_comm) {
        meta.sampling_comm = std::shared_ptr<MPI_Comm>(
            new MPI_Comm(MPI_COMM_NULL), [](MPI_Comm* comm) {
                int finalized = 0;
                MPI_Finalized(&finalized);
                if (!finalized && *comm != MPI_COMM_NULL) {
                    MPI_Comm_free(comm);
                }
                delete comm;
            });
        amrex::ParallelDescriptor::Comm_dup(
            amrex::ParallelDescriptor::Communicator(), *meta.sampling_comm);
    }
#endif
    int total_points = 0;
    for (auto& rotor : meta.rotors) {
        InitDataOp<ActuatorSector, ActSrcSector>()(rotor->data);
        total_points += static_cast<int>(rotor->data.grid().vel_pos.size());
    }
    const int n = meta.body_points.size();
    data.grid().resize(n, total_points + n);
    data.grid().epsilon.assign(n, vs::Vector::one() * meta.box_epsilon);
    data.grid().orientation.assign(n, vs::Tensor::identity());
    int offset = meta.box_points;
    for (const auto& arm : meta.arms) {
        std::copy(
            arm->grid().epsilon.begin(), arm->grid().epsilon.end(),
            data.grid().epsilon.begin() + offset);
        offset += arm->meta().num_pts;
    }
    UpdatePosOp<Drone, ActSrcDrone>()(data);
}

void UpdatePosOp<Drone, ActSrcDrone>::operator()(Drone::DataType& data)
{
    auto& grid = data.grid();
    int offset = 0;
    for (auto& rotor : data.meta().rotors) {
        UpdatePosOp<ActuatorSector, ActSrcSector>()(rotor->data);
        const auto& positions = rotor->data.grid().vel_pos;
        std::ranges::copy(positions, grid.vel_pos.begin() + offset);
        offset += static_cast<int>(positions.size());
    }
    const auto& m = data.meta();
    const auto t = data.sim().time().current_time() +
                   0.5_rt * sector::timestep_width(data.sim());
    const auto center = m.body_motion->position(t);
    const auto rotation = m.body_motion->orientation(t);
    for (int i = 0; i < static_cast<int>(m.body_points.size()); ++i) {
        grid.pos[i] = center + (rotation & m.body_points[i]);
        grid.vel_pos[offset + i] = grid.pos[i];
    }
}

void UpdateVelOp<Drone, ActSrcDrone>::operator()(Drone::DataType& data)
{
    auto& grid = data.grid();
    int offset = 0;
    for (auto& rotor : data.meta().rotors) {
        auto& rotor_grid = rotor->data.grid();
        const int npts = static_cast<int>(rotor_grid.vel.size());
        std::copy_n(grid.vel.begin() + offset, npts, rotor_grid.vel.begin());
        std::copy_n(
            grid.density.begin() + offset, npts, rotor_grid.density.begin());
        offset += npts;
    }
}

void ComputeForceOp<Drone, ActSrcDrone>::operator()(Drone::DataType& data)
{
    share_sampled_fields(data);
    if (data.info().root_proc >= 0 && !data.info().is_root_proc) {
        distribute_projection(data);
        return;
    }
    auto& meta = data.meta();
    const auto& time = data.sim().time();
    const amrex::Real midpoint_time =
        time.current_time() + (0.5_rt * sector::timestep_width(data.sim()));
    const auto drone_center = meta.body_motion->position(midpoint_time);
    meta.total_force = vs::Vector::zero();
    meta.total_moment = vs::Vector::zero();
    for (auto& rotor : meta.rotors) {
        ComputeForceOp<ActuatorSector, ActSrcSector>()(rotor->data);
        const auto& rotor_meta = rotor->data.meta();
        // Sector forces act on the fluid; report equal-and-opposite vehicle
        // load.
        meta.total_force = meta.total_force - rotor_meta.integrated_force;
        meta.total_moment =
            meta.total_moment - rotor_meta.integrated_moment -
            ((rotor_meta.center - drone_center) ^ rotor_meta.integrated_force);
    }
    auto& grid = data.grid();
    const auto rotation = meta.body_motion->orientation(midpoint_time);
    const auto translation =
        meta.body_motion->translation_velocity(midpoint_time);
    const auto omega = meta.body_motion->angular_velocity(midpoint_time);
    const int offset = grid.vel.size() - grid.pos.size();
    const auto relative_velocity = [&](int i) {
        return grid.vel[offset + i] - translation -
               (omega ^ (grid.pos[i] - drone_center));
    };
    for (int i = 0; i < meta.box_points; ++i) {
        const auto velocity = relative_velocity(i);
        grid.force[i] =
            -0.5_rt * meta.box_drag_area * vs::mag(velocity) * velocity;
    }
    int point = meta.box_points;
    for (int a = 0; a < static_cast<int>(meta.arms.size()); ++a) {
        auto& arm = *meta.arms[a];
        const auto frame = rotation & meta.arm_frames[a];
        for (int i = 0; i < arm.meta().num_pts; ++i)
            arm.grid().vel[i] = relative_velocity(point + i) & frame;
        UpdateVelOp<FixedWing, ActSrcLine>()(arm);
        ComputeForceOp<FixedWing, ActSrcLine>()(arm);
        for (int i = 0; i < arm.meta().num_pts; ++i, ++point) {
            grid.force[point] = frame & arm.grid().force[i];
            grid.orientation[point] =
                arm.grid().orientation[i] &
                vs::Tensor(frame.cx(), frame.cy(), frame.cz());
        }
    }
    for (int i = 0; i < static_cast<int>(grid.pos.size()); ++i) {
        meta.total_force = meta.total_force - grid.force[i];
        meta.total_moment =
            meta.total_moment - ((grid.pos[i] - drone_center) ^ grid.force[i]);
    }
    distribute_projection(data);
}

void ProcessOutputsOp<Drone, ActSrcDrone>::prepare_outputs(
    const std::string& out_dir)
{
    m_nc_filename = out_dir + "/" + m_data.info().label + ".nc";
    prepare_netcdf_file(m_nc_filename, m_data);
}

void ProcessOutputsOp<Drone, ActSrcDrone>::write_outputs()
{
    const auto& time = m_data.sim().time();
    if ((m_output_frequency > 0) &&
        (time.time_index() % m_output_frequency == 0)) {
        write_netcdf(m_nc_filename, m_data, time.new_time(), time.time_index());
    }
}

void ActSrcOp<Drone, ActSrcDrone>::initialize()
{
    m_points.resize(m_data.grid().pos.size());
    m_device_points.resize(m_points.size());
    int point = m_data.meta().box_points;
    for (const auto& arm : m_data.meta().arms) {
        for (int i = 0; i < arm->meta().num_pts; ++i)
            m_points[point++].dcoord_flags = arm->grid().dcoord_flags;
    }
    for (auto& rotor : m_data.meta().rotors) {
        rotor->source.initialize();
    }
}

void ActSrcOp<Drone, ActSrcDrone>::setup_op()
{
    if (m_data.info().root_proc >= 0 && !m_data.info().actuator_in_proc) {
        return;
    }
    const auto& grid = m_data.grid();
    for (int i = 0; i < static_cast<int>(m_points.size()); ++i) {
        auto& point = m_points[i];
        point.pos = grid.pos[i];
        point.force = grid.force[i];
        point.epsilon = grid.epsilon[i];
        point.orientation = grid.orientation[i];
    }
    m_structure_bounds = structure_bounds(m_data);
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice, m_points.begin(), m_points.end(),
        m_device_points.begin());
    for (auto& rotor : m_data.meta().rotors) {
        rotor->source.setup_op();
    }
}

void ActSrcOp<Drone, ActSrcDrone>::operator()(
    const int lev, const amrex::MFIter& mfi, const amrex::Geometry& geom)
{
    for (auto& rotor : m_data.meta().rotors) {
        rotor->source(lev, mfi, geom);
    }
    if (m_points.empty()) {
        return;
    }
    // Keep rotor evaluation above independent of structural support. Grow by
    // one cell to conservatively include support edges despite roundoff.
    auto support =
        ::kynema_sgf::utils::realbox_to_box(m_structure_bounds, geom);
    support.grow(1);
    const auto cells = mfi.tilebox() & support;
    if (cells.isEmpty()) {
        return;
    }
    // Drone positions already use the sector sampling time.
    const auto src =
        m_data.sim().repo().get_field("actuator_src_term")(lev).array(mfi);
    const auto lo = geom.ProbLoArray();
    const auto dx = geom.CellSizeArray();
    const int npts = static_cast<int>(m_device_points.size());
    const auto* points = m_device_points.data();
    amrex::ParallelFor(cells, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
        const vs::Vector x{
            lo[0] + ((i + 0.5_rt) * dx[0]), lo[1] + ((j + 0.5_rt) * dx[1]),
            lo[2] + ((k + 0.5_rt) * dx[2])};
        vs::Vector force = vs::Vector::zero();
        for (int ip = 0; ip < npts; ++ip) {
            const auto& p = points[ip];
            const auto dist = (p.orientation & (x - p.pos)) * p.dcoord_flags;
            force = force + utils::gaussian3d(dist, p.epsilon) * p.force;
        }
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            src(i, j, k, d) += force[d];
        }
    });
}

} // namespace kynema_sgf::actuator::ops
