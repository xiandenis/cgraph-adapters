#include "cloud/cloud_value.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace cloud {
namespace {

std::vector<float> json_to_floats(const nlohmann::json& arr) {
  if (!arr.is_array()) {
    throw std::invalid_argument("cloud: expected float array");
  }
  std::vector<float> out;
  out.reserve(arr.size());
  for (const auto& el : arr) {
    if (!el.is_number()) {
      throw std::invalid_argument("cloud: non-numeric coordinate");
    }
    out.push_back(el.get<float>());
  }
  return out;
}

}  // namespace

nlohmann::json make_xyz_cloud(const std::vector<float>& xyz_flat) {
  if (xyz_flat.size() % 3 != 0) {
    throw std::invalid_argument("cloud: xyz length must be multiple of 3");
  }
  nlohmann::json out = nlohmann::json::object();
  out["n"] = static_cast<std::int64_t>(xyz_flat.size() / 3);
  out["fields"] = nlohmann::json::array({"xyz"});
  out["xyz"] = xyz_flat;
  return out;
}

void require_cloud(const nlohmann::json& value) {
  if (!value.is_object() || !value.contains("n") || !value.contains("xyz") ||
      !value.contains("fields")) {
    throw std::invalid_argument("cloud: missing n/fields/xyz");
  }
  if (!value["n"].is_number_integer()) {
    throw std::invalid_argument("cloud: n must be integer");
  }
  const auto n = static_cast<std::size_t>(value["n"].get<std::int64_t>());
  const auto xyz = json_to_floats(value["xyz"]);
  if (xyz.size() != n * 3) {
    throw std::invalid_argument("cloud: xyz size != 3n");
  }
  if (value.contains("normal")) {
    const auto normal = json_to_floats(value["normal"]);
    if (normal.size() != n * 3) {
      throw std::invalid_argument("cloud: normal size != 3n");
    }
  }
  if (value.contains("feat")) {
    if (!value.contains("feat_dim") || !value["feat_dim"].is_number_integer()) {
      throw std::invalid_argument("cloud: feat requires feat_dim");
    }
    const auto d = static_cast<std::size_t>(value["feat_dim"].get<std::int64_t>());
    const auto feat = json_to_floats(value["feat"]);
    if (feat.size() != n * d) {
      throw std::invalid_argument("cloud: feat size != D*n");
    }
  }
}

std::size_t cloud_n(const nlohmann::json& value) {
  require_cloud(value);
  return static_cast<std::size_t>(value["n"].get<std::int64_t>());
}

}  // namespace cloud
