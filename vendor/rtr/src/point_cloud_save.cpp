#include "rtr/cloud_file_util.hpp"
#include "rtr/ops.hpp"

#include <filesystem>
#include <memory>

namespace rtr {
namespace {

class PointCloudSaveOp final : public cgraph::MemoryOperator {
 public:
  PointCloudSaveOp() {
    op_id_ = "rtr.point_cloud.save";
    signature_.inputs["cloud"] = cloud_port("cloud");
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["cloud"] = cloud_port("cloud");
    signature_.params["format"] = cloud_format_param();
    capability_.summary =
        "Write a point cloud to a user path in las/pcd/ply (default las)";
    cost_.cost_class = "cpu.medium";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "读入工作区点云文件，按 format（默认 las）写出到用户指定 path。"
        "path 扩展名必须与 format 一致（.las/.pcd/.ply）；不支持的 format 直接报错。"
        "这是唯一允许用户指定点云写出路径的算子。";
    usage_.notes = {"经编解码写出，非纯复制", "允许 format: las, pcd, ply"};
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("cloud") || !inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.save: missing cloud/path");
    }
    if (inputs.at("path").payload.kind() != cgraph::Payload::Kind::String) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.save: path must be string");
    }
    const CloudFileFormat fmt =
        parse_cloud_format(params, "rtr.point_cloud.save");
    require_file_point_cloud(inputs.at("cloud"), "rtr.point_cloud.save");
    auto cloud = load_xyz_cloud(inputs.at("cloud"), "rtr.point_cloud.save");
    const auto dest = std::filesystem::path(inputs.at("path").payload.as_string());
    if (dest.empty() || dest.filename().empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.save: path must include a file name");
    }
    require_path_matches_format(dest, fmt, "rtr.point_cloud.save");
    return {{"cloud", save_xyz_cloud_format(*cloud, dest, "rtr.point_cloud.save")}};
  }
};

}  // namespace

void register_point_cloud_save(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<PointCloudSaveOp>());
}

}  // namespace rtr
