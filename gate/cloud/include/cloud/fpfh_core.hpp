#pragma once

#include <nlohmann/json.hpp>

namespace cloud {

/** Shared FPFH gate implementation (Memory and fake Process). */
nlohmann::json compute_fpfh(const nlohmann::json& cloud_with_normals, double radius_f);

}  // namespace cloud
