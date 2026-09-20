#include "rtr/cloud_file_util.hpp"
#include "rtr/json.hpp"
#include "rtr/ops.hpp"

#include <registration_type/lidar_frame_io.hpp>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace rtr {
namespace {

class LidarFrameLoadOp final : public cgraph::MemoryOperator {
 public:
  LidarFrameLoadOp() {
    op_id_ = "rtr.lidar_frame.load";
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["frame"] = lidar_frame_port("frame");
    signature_.outputs["file"] = lidar_frame_file_port("file");
    capability_.summary = "Load LidarFrame JSON into isomorphic Value + file Artifact";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "读取 RTR LidarFrame 本地化 JSON，吐出同构内存 Value（无点数组）与旁路 "
        "lidar_frame_file Artifact。不因 load Frame 而自动物化点云 Buffer。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.lidar_frame.load: missing path");
    }
    const auto path = std::filesystem::path(inputs.at("path").payload.as_string());
    if (!std::filesystem::is_regular_file(path)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.lidar_frame.load: file missing");
    }
    Ddx::LidarFrame frame;
    if (!Ddx::LidarFrameIO::loadLidarFrame(path.string(), frame)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.lidar_frame.load: loadLidarFrame failed");
    }
    return {
        {"frame", lidar_frame_to_data(frame)},
        {"file", file_artifact_from_path(path, "rtr.type.lidar_frame_file",
                                         "rtr.semantic.lidar_frame_file")},
    };
  }
};

class LidarFrameSaveOp final : public cgraph::MemoryOperator {
 public:
  LidarFrameSaveOp() {
    op_id_ = "rtr.lidar_frame.save";
    signature_.inputs["frame"] = lidar_frame_port("frame");
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["file"] = lidar_frame_file_port("file");
    capability_.summary = "Write isomorphic LidarFrame Value via LidarFrameIO";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "将同构 lidar_frame Value 写出为 RTR 本地化 JSON。"
        "不序列化内存点云；路径字段指向已有点云文件。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("frame") || !inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.lidar_frame.save: missing frame/path");
    }
    const auto out_path = std::filesystem::path(inputs.at("path").payload.as_string());
    std::filesystem::create_directories(out_path.parent_path());
    const Ddx::LidarFrame frame = lidar_frame_from_data(inputs.at("frame"));
    if (!Ddx::LidarFrameIO::saveLidarFrame(out_path.string(), frame)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.lidar_frame.save: saveLidarFrame failed");
    }
    return {{"file", file_artifact_from_path(out_path, "rtr.type.lidar_frame_file",
                                             "rtr.semantic.lidar_frame_file")}};
  }
};

}  // namespace

void register_lidar_frame_io(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<LidarFrameLoadOp>());
  registry.add(std::make_shared<LidarFrameSaveOp>());
}

}  // namespace rtr
