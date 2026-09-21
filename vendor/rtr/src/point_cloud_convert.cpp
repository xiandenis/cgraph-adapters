#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <memory>

namespace rtr {
namespace {

class PointCloudConvertOp final : public cgraph::MemoryOperator {
 public:
  PointCloudConvertOp() {
    op_id_ = "rtr.point_cloud.convert";
    signature_.inputs["cloud"] = cloud_port("cloud");
    signature_.outputs["cloud"] = cloud_port("cloud");
    signature_.params["format"] = cloud_format_param();
    capability_.summary =
        "Convert a point-cloud file among las/pcd/ply (writes graph workspace)";
    cost_.cost_class = "cpu.medium";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "把点云文件在 las / pcd / ply 之间转换，结果写入图工作区 "
        "outputs/cloud.<ext>。路径不可由用户指定；需要自定义目录请用 "
        "rtr.point_cloud.save。";
    usage_.notes = {"允许 format: las, pcd, ply", "默认 format=las"};
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    if (!inputs.count("cloud")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.convert: missing cloud");
    }
    const CloudFileFormat fmt =
        parse_cloud_format(params, "rtr.point_cloud.convert");
    require_file_point_cloud(inputs.at("cloud"), "rtr.point_cloud.convert");
    auto cloud = load_xyz_cloud(inputs.at("cloud"), "rtr.point_cloud.convert");
    const auto out_path = workspace_cloud_path(ctx, "cloud", fmt);
    return {{"cloud",
             save_xyz_cloud_format(*cloud, out_path, "rtr.point_cloud.convert")}};
  }
};

}  // namespace

void register_point_cloud_convert(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<PointCloudConvertOp>());
}

}  // namespace rtr
