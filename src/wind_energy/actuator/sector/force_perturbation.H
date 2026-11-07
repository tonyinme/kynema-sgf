#ifndef SECTOR_FORCE_PERTURBATION_H
#define SECTOR_FORCE_PERTURBATION_H

#include <cstdint>
#include <initializer_list>
#include <string_view>

namespace kynema_sgf::actuator::sector {

// SplitMix64 finalizer; unsigned arithmetic is intentional and reproducible.
inline std::uint64_t perturbation_hash(std::uint64_t x)
{
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

inline std::uint64_t perturbation_label(const std::string_view label)
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char c : label) {
        hash = (hash ^ c) * UINT64_C(1099511628211);
    }
    return hash;
}

// Uniform [-1, 1); independent of MPI rank and repeated calls within a step.
inline double force_perturbation(
    std::uint64_t key,
    const int seed,
    const int rotor,
    const int blade,
    const int radial,
    const int swept,
    const int step)
{
    for (const int id : {seed, rotor, blade, radial, swept, step}) {
        key = perturbation_hash(
            key + static_cast<std::uint64_t>(id) +
            UINT64_C(0x9e3779b97f4a7c15));
    }
    return 2.0 * static_cast<double>(key >> 11) * 0x1.0p-53 - 1.0;
}

} // namespace kynema_sgf::actuator::sector
#endif
