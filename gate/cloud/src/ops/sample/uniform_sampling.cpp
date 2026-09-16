#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cloud {
namespace {

class UniformSamplingOp final : public cgraph::MemoryOperator {
 public:
  UniformSamplingOp() {
    op_id_ = "cloud.uniform_sampling";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.inputs["in"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
    signature_.outputs["out"].formats = {"native"};
    signature_.outputs["out"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    cgraph::ParamSpec radius;
    radius.name = "radius";
    radius.dtype = "float";
    radius.default_value = 0.05;
    radius.invalidate = true;
    signature_.params["radius"] = std::move(radius);
    capability_.summary = "Radius-based uniform sampling (gate; not VoxelGrid)";
    cost_.cost_class = "cpu.n2";
    usage_.connect = "Wire cloud{xyz} into in; read thinned cloud from out.";
    usage_.tune = "params.radius > 0: greedy keep-first within radius.";
    usage_.inspect = "out.n <= in.n; distinct from cloud.voxel leaf grid.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.uniform_sampling: missing input 'in'");
    }
    require_cloud(it->second);
    double radius = 0.05;
    if (params.is_object() && params.contains("radius") && params["radius"].is_number()) {
      radius = params["radius"].get<double>();
    }
    if (radius <= 0.0) {
      return {{"out", it->second}};
    }
    const double r2 = radius * radius;
    const auto n = cloud_n(it->second);
    const auto& xyz_json = it->second["xyz"];
    std::vector<float> kept;
    kept.reserve(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
      const float x = xyz_json[3 * i].get<float>();
      const float y = xyz_json[3 * i + 1].get<float>();
      const float z = xyz_json[3 * i + 2].get<float>();
      bool ok = true;
      for (std::size_t j = 0; j < kept.size(); j += 3) {
        const double dx = static_cast<double>(x) - kept[j];
        const double dy = static_cast<double>(y) - kept[j + 1];
        const double dz = static_cast<double>(z) - kept[j + 2];
        if (dx * dx + dy * dy + dz * dz < r2) {
          ok = false;
          break;
        }
      }
      if (ok) {
        kept.push_back(x);
        kept.push_back(y);
        kept.push_back(z);
      }
    }
    return {{"out", make_xyz_cloud(kept)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_uniform_sampling() {
  return std::make_shared<UniformSamplingOp>();
}

}  // namespace cloud
