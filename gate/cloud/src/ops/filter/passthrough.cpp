#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"
#include "cloud/usage_helpers.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cloud {
namespace {

class PassThroughOp final : public cgraph::MemoryOperator {
 public:
  PassThroughOp() {
    op_id_ = "cloud.passthrough";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.inputs["in"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
    signature_.outputs["out"].formats = {"native"};
    signature_.outputs["out"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    cgraph::ParamSpec axis;
    axis.name = "axis";
    axis.dtype = "string";
    axis.default_value = "z";
    axis.invalidate = true;
    signature_.params["axis"] = std::move(axis);
    cgraph::ParamSpec min_v;
    min_v.name = "min";
    min_v.dtype = "float";
    min_v.default_value = -1.0e30;
    min_v.invalidate = true;
    signature_.params["min"] = std::move(min_v);
    cgraph::ParamSpec max_v;
    max_v.name = "max";
    max_v.dtype = "float";
    max_v.default_value = 1.0e30;
    max_v.invalidate = true;
    signature_.params["max"] = std::move(max_v);
    capability_.summary = "Axis range crop (PassThrough gate)";
    cost_.cost_class = "cpu.n";
    usage_.connect = "Wire cloud{xyz} into in; read cropped cloud from out.";
    usage_.tune = "params.axis in {x,y,z}; keep points with min <= coord <= max.";
    usage_.inspect = "out.n <= in.n; fields remain [xyz].";
    usage_.for_whom = "按 x/y/z 轴范围裁剪点云";
    usage_.min_graph = cloud::usage::make_min_graph_node(
        "crop", "cloud.passthrough", {{"axis", "z"}, {"min", 0.0}, {"max", 1.0}},
        {{"in", cloud::usage::cloud_in_xyz()}},
        {{"out", cloud::usage::cloud_out_xyz()}});
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.passthrough: missing input 'in'");
    }
    require_cloud(it->second);
    std::string axis = "z";
    double min_v = -1.0e30;
    double max_v = 1.0e30;
    if (params.is_object()) {
      if (params.contains("axis") && params["axis"].is_string()) {
        axis = params["axis"].get<std::string>();
      }
      if (params.contains("min") && params["min"].is_number()) {
        min_v = params["min"].get<double>();
      }
      if (params.contains("max") && params["max"].is_number()) {
        max_v = params["max"].get<double>();
      }
    }
    int axis_i = 2;
    if (axis == "x") {
      axis_i = 0;
    } else if (axis == "y") {
      axis_i = 1;
    } else if (axis == "z") {
      axis_i = 2;
    } else {
      throw std::invalid_argument("cloud.passthrough: axis must be x|y|z");
    }
    const auto n = cloud_n(it->second);
    const auto& xyz_json = it->second["xyz"];
    std::vector<float> out_xyz;
    out_xyz.reserve(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
      const float x = xyz_json[3 * i].get<float>();
      const float y = xyz_json[3 * i + 1].get<float>();
      const float z = xyz_json[3 * i + 2].get<float>();
      const float c = axis_i == 0 ? x : (axis_i == 1 ? y : z);
      if (static_cast<double>(c) >= min_v && static_cast<double>(c) <= max_v) {
        out_xyz.push_back(x);
        out_xyz.push_back(y);
        out_xyz.push_back(z);
      }
    }
    return {{"out", make_xyz_cloud(out_xyz)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_passthrough() {
  return std::make_shared<PassThroughOp>();
}

}  // namespace cloud
