#include "rtr/json.hpp"
#include "rtr/ops.hpp"

#include <point_cloud_io/point_cloud_io.hpp>

#include "cgraph/ops.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <filesystem>
#include <memory>
#include <string>

namespace rtr {
namespace {

class PointCloudLoadOp final : public cgraph::MemoryOperator {
 public:
  PointCloudLoadOp() {
    op_id_ = "rtr.point_cloud.load";
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["cloud"] = cloud_port("cloud");
    capability_.summary =
        "Validate point-cloud path and emit rtr.type.point_cloud Artifact (file ref)";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "将磁盘上的点云路径校验为可读文件引用（Artifact）。不把点云抬进图内存；"
        "后续算子按路径自行载入。支持 PCD / PLY / LAS / SLAS。";
    usage_.notes = {"失败不返回空 Artifact", "输出 URI 为规范化绝对路径"};
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.load: missing path");
    }
    if (inputs.at("path").payload.kind() != cgraph::Payload::Kind::String) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.load: path must be string");
    }
    const std::filesystem::path path = inputs.at("path").payload.as_string();
    if (!std::filesystem::is_regular_file(path)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.load: file missing");
    }
    const auto fmt =
        Ddx::point_cloud_io::detect_format(path.string());
    if (fmt == Ddx::point_cloud_io::PointCloudFormat::Unknown) {
      throw cgraph::OperatorError(
          cgraph::ErrorCode::OpFailed,
          "rtr.point_cloud.load: unsupported extension (need pcd/ply/las/slas)");
    }
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (Ddx::point_cloud_io::load(path.string(), cloud) != 0 || cloud.empty()) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.point_cloud.load: load() failed or empty");
    }
    return {{"cloud", cloud_artifact_from_path(path)}};
  }
};

}  // namespace

void register_point_cloud_load(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<PointCloudLoadOp>());
}

}  // namespace rtr
