#pragma once

#include <nlohmann/json.hpp>
#include <vector>

namespace cloud {

/** JSON rigid_transform: {"R":[[3x3 row-major]], "t":[3]} */
nlohmann::json make_identity_rigid();
nlohmann::json make_rigid(const std::vector<double>& R9, const std::vector<double>& t3);
void require_rigid(const nlohmann::json& value);

/** JSON correspondence_set: {"pairs":[[i,j], ...]} */
nlohmann::json make_correspondences(const std::vector<std::pair<int, int>>& pairs);
void require_correspondences(const nlohmann::json& value);

/**
 * Absolute orientation (Kabsch/SVD) from equal-length point lists.
 * src/tgt: flat xyz float triples (same count).
 * Returns rigid_transform JSON.
 */
nlohmann::json kabsch_rigid(const std::vector<float>& src_xyz,
                            const std::vector<float>& tgt_xyz);

}  // namespace cloud
