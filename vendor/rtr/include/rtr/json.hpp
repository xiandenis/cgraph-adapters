#pragma once

#include <cgraph/artifact.hpp>
#include <registration_type/registration_type.hpp>

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace rtr {

inline void throw_json(const std::string& msg) {
  throw std::invalid_argument(msg);
}

nlohmann::json matrix4d_to_json(const Eigen::Matrix4d& m);
Eigen::Matrix4d matrix4d_from_json(const nlohmann::json& j);
nlohmann::json matrix6d_to_json(const Eigen::Matrix<double, 6, 6>& m);
Eigen::Matrix<double, 6, 6> matrix6d_from_json(const nlohmann::json& j);
nlohmann::json align_result_to_json(const Ddx::AlignResult& a);
Ddx::AlignResult align_result_from_json(const nlohmann::json& j);
std::filesystem::path artifact_file_path(const nlohmann::json& handle);

}  // namespace rtr
