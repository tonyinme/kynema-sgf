#ifndef DRONE_OPS_H
#define DRONE_OPS_H

#include "src/wind_energy/actuator/drone/Drone.H"
#include "src/wind_energy/actuator/actuator_ops.H"

namespace kynema_sgf::actuator::ops {

template <>
void determine_root_proc<Drone>(
    Drone::DataType& data, amrex::Vector<int>& act_proc_count);

template <>
void determine_influenced_procs<Drone>(Drone::DataType& data);

template <>
struct ReadInputsOp<Drone, ActSrcDrone>
{
    void operator()(Drone::DataType& data, const utils::ActParser& pp);
};

template <>
struct InitDataOp<Drone, ActSrcDrone>
{
    void operator()(Drone::DataType& data);
};

template <>
struct UpdatePosOp<Drone, ActSrcDrone>
{
    void operator()(Drone::DataType& data);
};

template <>
struct UpdateVelOp<Drone, ActSrcDrone>
{
    void operator()(Drone::DataType& data);
};

template <>
struct ComputeForceOp<Drone, ActSrcDrone>
{
    void operator()(Drone::DataType& data);
};

template <>
struct RefinementGeometryOp<Drone, ActSrcDrone>
{
    [[nodiscard]] amrex::Vector<ActuatorRefinementGeometry>
    operator()(const Drone::DataType& data, amrex::Real time) const;
};

template <>
struct ReferenceFrameOp<Drone, ActSrcDrone>
{
    [[nodiscard]] std::optional<motion::RigidTransform>
    operator()(const Drone::DataType& data, amrex::Real time) const;
};

template <>
struct ProcessOutputsOp<Drone, ActSrcDrone>
{
    explicit ProcessOutputsOp(Drone::DataType& data) : m_data(data) {}
    void read_io_options(const utils::ActParser& pp)
    {
        pp.query("output_frequency", m_output_frequency);
    }
    void prepare_outputs(const std::string& out_dir);
    void write_outputs();

private:
    Drone::DataType& m_data;
    std::string m_nc_filename;
    int m_output_frequency{1};
};

template <>
class ActSrcOp<Drone, ActSrcDrone>
{
public:
    explicit ActSrcOp(Drone::DataType& data) : m_data(data) {}
    void initialize();
    void setup_op();
    void
    operator()(int lev, const amrex::MFIter& mfi, const amrex::Geometry& geom);

private:
    Drone::DataType& m_data;
    struct SourcePoint
    {
        vs::Vector pos, force, epsilon;
        vs::Vector dcoord_flags{1, 1, 1};
        vs::Tensor orientation;
    };
    // Rebuilt each setup from the existing 4-epsilon structural support.
    amrex::RealBox m_structure_bounds;
    amrex::Vector<SourcePoint> m_points;
    amrex::Gpu::DeviceVector<SourcePoint> m_device_points;
};

} // namespace kynema_sgf::actuator::ops

#endif
