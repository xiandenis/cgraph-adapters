#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/usage_helpers.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cloud {
namespace {

class VoxelOp final : public cgraph::MemoryOperator {
 public:
  VoxelOp() {
    op_id_ = "cloud.voxel";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.inputs["in"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
    signature_.outputs["out"].formats = {"native"};
    signature_.outputs["out"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    cgraph::ParamSpec leaf;
    leaf.name = "leaf";
    leaf.dtype = "float";
    leaf.default_value = 0.05;
    leaf.invalidate = true;
    signature_.params["leaf"] = std::move(leaf);
    capability_.summary = "Uniform voxel downsample (xyz only, Step 11 stub)";
    capability_.tags = {"downsample", "voxel"};
    cost_.cost_class = "cpu.n";
    usage_.connect = "Wire cloud{xyz} into in; read downsampled cloud from out.";
    usage_.tune = "params.leaf > 0 sets grid size; leaf <= 0 keeps all points.";
    usage_.inspect = "out.n <= in.n; fields remain [xyz].";
    usage_.for_whom = "点太多、后面特征/配准太慢，还不懂体素的人";
    usage_.connect_bullets = {
        "把 Load（或上一处理）的 cloud 接到输入 in",
        "输出 cloud 再往下接；不要指望它长出法向",
    };
    usage_.set_bullets = {
        "leaf 单位与点云一致，一般是米",
        "量相邻点距 d，leaf 先取 2d～5d；量不了就用 0.05",
    };
    usage_.look_bullets = {
        "输出点数应明显少于输入",
        "物体轮廓还在，不是空云",
    };
    usage_.first_values.push_back(
        cgraph::UsageFirstValue{"leaf", 0.05, "相对点间距 2d～5d，默认 0.05"});
    cloud::usage::set_tune(usage_, {
        {"点数几乎没变", "leaf", "leaf ×2"},
        {"棱被切圆、配准对不齐", "leaf", "leaf ÷2"},
        {"下游报缺 normal", "—", "接 cloud.normal_est"},
    });
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "voxel", "cloud.voxel", {{"leaf", 0.05}},
        {{"in", cloud::usage::cloud_in_xyz()}},
        {{"out", cloud::usage::cloud_out_xyz()}});
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.voxel: missing input 'in'");
    }
    require_cloud(it->second);
    double leaf = 0.05;
    if (params.is_object() && params.contains("leaf") && params["leaf"].is_number()) {
      leaf = params["leaf"].get<double>();
    }
    if (leaf <= 0.0) {
      return {{"out", it->second}};
    }
    const auto n = cloud_n(it->second);
    const auto& xyz_json = it->second["xyz"];
    std::vector<float> out_xyz;
    out_xyz.reserve(n * 3);
    std::unordered_set<std::int64_t> seen;
    seen.reserve(n);
    const double inv = 1.0 / leaf;
    for (std::size_t i = 0; i < n; ++i) {
      const float x = xyz_json[3 * i].get<float>();
      const float y = xyz_json[3 * i + 1].get<float>();
      const float z = xyz_json[3 * i + 2].get<float>();
      const auto ix = static_cast<std::int64_t>(std::floor(x * inv));
      const auto iy = static_cast<std::int64_t>(std::floor(y * inv));
      const auto iz = static_cast<std::int64_t>(std::floor(z * inv));
      // Pack 21-bit axes into one key (enough for synth fixtures).
      const std::int64_t key = ((ix & 0x1FFFFF) << 42) | ((iy & 0x1FFFFF) << 21) |
                               (iz & 0x1FFFFF);
      if (seen.insert(key).second) {
        out_xyz.push_back(x);
        out_xyz.push_back(y);
        out_xyz.push_back(z);
      }
    }
    return {{"out", make_xyz_cloud(out_xyz)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_voxel() {
  return std::make_shared<VoxelOp>();
}

}  // namespace cloud
