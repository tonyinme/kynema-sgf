#ifndef ACTUATOR_SECTOR_SECTION_AERODYNAMICS_H
#define ACTUATOR_SECTOR_SECTION_AERODYNAMICS_H

#include <cmath>

namespace kynema_sgf::actuator::sector {

/**
 * Propeller incidence in the sector basis (e_r x e_theta = -e_normal).
 *
 * A positive effective pitch produces blade lift toward -e_normal. Thus
 * flow toward +e_normal reduces incidence. Mirroring a propeller reverses
 * both omega and signed_twist_radians, preserving effective pitch.
 */
template <typename Scalar>
inline Scalar section_angle_of_attack(
    const Scalar vtheta,
    const Scalar vnormal,
    const Scalar signed_twist_radians,
    const Scalar omega)
{
    const Scalar spin_sign = (omega >= Scalar(0)) ? Scalar(1) : Scalar(-1);
    const Scalar inflow = std::atan2(vnormal, -spin_sign * vtheta);
    const Scalar two_pi = Scalar(2) * std::acos(Scalar(-1));
    return std::remainder(spin_sign * signed_twist_radians - inflow, two_pi);
}

/** Coefficients of the Gaussian chord/thickness axes in (e_theta, e_normal). */
template <typename Scalar>
struct SectionAxisCoefficients
{
    Scalar chord_theta;
    Scalar chord_normal;
    Scalar thickness_theta;
    Scalar thickness_normal;
};

template <typename Scalar>
inline SectionAxisCoefficients<Scalar>
section_axis_coefficients(const Scalar signed_twist_radians)
{
    const Scalar cosine = std::cos(signed_twist_radians);
    const Scalar sine = std::sin(signed_twist_radians);
    // The leading-edge-directed chord is spin_sign times this chord axis.
    return {cosine, -sine, sine, cosine};
}

} // namespace kynema_sgf::actuator::sector

#endif
