#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <resolution_estimate/resolution_estimate.hpp>

#include <cmath>
#include <memory>

namespace rtr {
namespace {

class ResolutionEstimateOp final : public cgraph::MemoryOperator {
 public:
  ResolutionEstimateOp() {
    op_id_ = "rtr.resolution_estimate";
    signature_.inputs["cloud"] = cloud_port("cloud");
    signature_.outputs["voxel_size"] = scalar_float_port("voxel_size");
    capability_.summary = "Estimate registration voxel size (RTR ResolutionEstimate)";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "从点云文件估计配准用体素尺度（分辨率估计），输出 float voxel_size。"
        "只读 File realisation，不接受内存 Buffer，不写点云。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("cloud")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.resolution_estimate: missing cloud");
    }
    require_file_point_cloud(inputs.at("cloud"), "rtr.resolution_estimate");
    auto cloud = load_xyz_cloud(inputs.at("cloud"), "rtr.resolution_estimate");
    const float vs = Ddx::ResolutionEstimate::estimate(cloud, "cgraph");
    if (!(vs > 0.f) || !std::isfinite(vs)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.resolution_estimate: estimate failed");
    }
    return {{"voxel_size",
             cgraph::make_data_object(cgraph::type_ids::floating(),
                                      cgraph::SemanticSpec::of("rtr.semantic.voxel_size"),
                                      cgraph::Payload::floating(static_cast<double>(vs)))}};
  }
};

}  // namespace

void register_resolution_estimate(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<ResolutionEstimateOp>());
}

}  // namespace rtr
