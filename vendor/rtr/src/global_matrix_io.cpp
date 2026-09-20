#include "rtr/cloud_file_util.hpp"
#include "rtr/json.hpp"
#include "rtr/ops.hpp"

#include <global_matrix_file/global_matrix_file_io.hpp>

#include <filesystem>
#include <map>
#include <memory>
#include <string>

namespace rtr {
namespace {

class GlobalMatrixLoadOp final : public cgraph::MemoryOperator {
 public:
  GlobalMatrixLoadOp() {
    op_id_ = "rtr.global_matrix.load";
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["table"] = global_matrix_table_port("table");
    signature_.outputs["file"] = global_matrix_file_port("file");
    capability_.summary = "Load station global matrices JSON (GlobalMatrixFileIO)";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "读取 RTR global_matrix 本地化 JSON（站名→全局 4×4），吐出 station_global_pose 列表。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.global_matrix.load: missing path");
    }
    const auto path = std::filesystem::path(inputs.at("path").payload.as_string());
    if (!std::filesystem::is_regular_file(path)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.global_matrix.load: file missing");
    }
    std::map<std::string, Eigen::Matrix4d> table;
    if (!Ddx::GlobalMatrixFileIO::loading(path.string(), table)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.global_matrix.load: loading failed");
    }
    return {{"table", global_matrix_table_to_data(table)},
            {"file", file_artifact_from_path(path, "rtr.type.global_matrix_file",
                                             "rtr.semantic.global_matrix_file")}};
  }
};

class GlobalMatrixSaveOp final : public cgraph::MemoryOperator {
 public:
  GlobalMatrixSaveOp() {
    op_id_ = "rtr.global_matrix.save";
    signature_.inputs["table"] = global_matrix_table_port("table");
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["file"] = global_matrix_file_port("file");
    capability_.summary = "Save station global matrices JSON (GlobalMatrixFileIO)";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "把站名→全局 4×4 表本地化写出为 RTR global_matrix JSON。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("table") || !inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.global_matrix.save: missing table/path");
    }
    const auto path = std::filesystem::path(inputs.at("path").payload.as_string());
    std::filesystem::create_directories(path.parent_path());
    const auto table = global_matrix_table_from_data(inputs.at("table"));
    if (!Ddx::GlobalMatrixFileIO::saving(path.string(), table)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.global_matrix.save: saving failed");
    }
    return {{"file", file_artifact_from_path(path, "rtr.type.global_matrix_file",
                                             "rtr.semantic.global_matrix_file")}};
  }
};

}  // namespace

void register_global_matrix_io(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<GlobalMatrixSaveOp>());
  registry.add(std::make_shared<GlobalMatrixLoadOp>());
}

}  // namespace rtr
