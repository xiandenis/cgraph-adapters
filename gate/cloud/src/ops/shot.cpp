#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cloud {
namespace {

constexpr int kShotFeatDim = 352;

nlohmann::json compute_shot_stub(const nlohmann::json& cloud_with_normals,
                                 double radius_shot) {
  require_cloud(cloud_with_normals);
  if (!cloud_with_normals.contains("normal")) {
    throw std::invalid_argument("cloud.shot: requires normal field");
  }
  const std::size_t n = cloud_n(cloud_with_normals);
  const auto& xyz = cloud_with_normals["xyz"];
  const auto& normals = cloud_with_normals["normal"];
  if (!xyz.is_array() || !normals.is_array()) {
    throw std::invalid_argument("cloud.shot: bad xyz/normal");
  }
  std::vector<float> feat(n * static_cast<std::size_t>(kShotFeatDim), 0.0f);
  const float inv_r = radius_shot > 1e-12 ? static_cast<float>(1.0 / radius_shot) : 1.0f;
  for (std::size_t i = 0; i < n; ++i) {
    const float nx = normals[3 * i].get<float>();
    const float ny = normals[3 * i + 1].get<float>();
    const float nz = normals[3 * i + 2].get<float>();
    const float x = xyz[3 * i].get<float>() * inv_r;
    const float y = xyz[3 * i + 1].get<float>() * inv_r;
    const float z = xyz[3 * i + 2].get<float>() * inv_r;
    float* row = feat.data() + i * static_cast<std::size_t>(kShotFeatDim);
    // Deterministic gate stub: pack local frame stats into fixed 352 bins (not PCL SHOT).
    for (int b = 0; b < kShotFeatDim; ++b) {
      const float phase = static_cast<float>(b) * 0.017453292f;
      row[b] = 0.5f + 0.5f * std::tanh(nx * std::cos(phase) + ny * std::sin(phase) +
                                        nz * std::cos(2.0f * phase) +
                                        0.1f * (x + y + z) * std::sin(phase));
    }
  }
  nlohmann::json out = cloud_with_normals;
  out["feat"] = feat;
  out["feat_dim"] = kShotFeatDim;
  out["fields"] = nlohmann::json::array({"xyz", "normal", "feat"});
  return out;
}

class ShotOp final : public cgraph::MemoryOperator {
 public:
  ShotOp() {
    op_id_ = "cloud.shot";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.inputs["in"].schema = {{"require", nlohmann::json::array({"xyz", "normal"})}};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
    signature_.outputs["out"].formats = {"native"};
    signature_.outputs["out"].schema = {
        {"fields", nlohmann::json::array({"xyz", "normal", "feat"})},
        {"feat", kShotFeatDim},
    };
    cgraph::ParamSpec radius;
    radius.name = "radius_shot";
    radius.dtype = "float";
    radius.default_value = 0.2;
    radius.invalidate = true;
    signature_.params["radius_shot"] = std::move(radius);
    capability_.summary = "SHOT feat[352] gate stub (not PCL)";
    cost_.cost_class = "cpu.n_k";
    usage_.connect = "Require cloud{xyz,normal}; out has feat_dim=352.";
    usage_.tune = "params.radius_shot neighborhood scale (gate stub).";
    usage_.inspect = "feat length = 352 * n; gate only, not production SHOT.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.shot: missing in");
    }
    double radius_shot = 0.2;
    if (params.is_object() && params.contains("radius_shot") &&
        params["radius_shot"].is_number()) {
      radius_shot = params["radius_shot"].get<double>();
    }
    return {{"out", compute_shot_stub(it->second, radius_shot)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_shot() {
  auto op = std::make_shared<ShotOp>();
  op->set_impl_fingerprint("shot-gate-stub-v1");
  return op;
}

}  // namespace cloud
