#include "rtr/cloud_file_util.hpp"
#include "rtr/json.hpp"
#include "rtr/ops.hpp"

#include "cgraph/ops.hpp"

#include <filesystem>
#include <memory>
#include <string>

namespace rtr {
namespace {

class PointCloudToBufferOp final : public cgraph::MemoryOperator {
 public:
  PointCloudToBufferOp() {
    op_id_ = "rtr.point_cloud.to_buffer";
    signature_.inputs["cloud"] = cloud_port("cloud");
    signature_.outputs["cloud"] = cloud_buffer_port("cloud");
    capability_.summary = "Materialize File point_cloud into Buffer opaque bytes";
    cost_.cost_class = "cpu.heavy";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "显式 File→Buffer 物化：读盘并用 rtr.codec.point_xyz_f32 编码。"
        "副作用：磁盘读。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("cloud")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.to_buffer: missing cloud");
    }
    auto cloud = load_xyz_cloud(inputs.at("cloud"), "rtr.point_cloud.to_buffer");
    return {{"cloud", point_cloud_buffer_from_xyz(*cloud)}};
  }
};

class PointCloudToFileOp final : public cgraph::MemoryOperator {
 public:
  PointCloudToFileOp() {
    op_id_ = "rtr.point_cloud.to_file";
    signature_.inputs["cloud"] = cloud_buffer_port("cloud");
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["cloud"] = cloud_port("cloud");
    capability_.summary = "Write Buffer point_cloud to PCD Artifact";
    cost_.cost_class = "cpu.heavy";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "显式 Buffer→File 物化：写旁路 PCD 并吐出 File realisation Artifact。"
        "副作用：磁盘写。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("cloud") || !inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.to_file: missing cloud/path");
    }
    auto cloud = load_xyz_cloud(inputs.at("cloud"), "rtr.point_cloud.to_file");
    const auto out_path =
        std::filesystem::path(inputs.at("path").payload.as_string());
    return {{"cloud", save_xyz_cloud_artifact(*cloud, out_path,
                                              "rtr.point_cloud.to_file")}};
  }
};

}  // namespace

void register_point_cloud_materialize(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<PointCloudToBufferOp>());
  registry.add(std::make_shared<PointCloudToFileOp>());
}

}  // namespace rtr
