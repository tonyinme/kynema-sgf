#ifndef ACTUATOR_SECTOR_SECTION_AERODYNAMICS_H
#define ACTUATOR_SECTOR_SECTION_AERODYNAMICS_H

#include <cmath>

namespace kynema_sgf::actuator::sector {

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
