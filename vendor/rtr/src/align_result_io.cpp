#include "rtr/cloud_file_util.hpp"
#include "rtr/json.hpp"
#include "rtr/ops.hpp"

#include <registration_type/align_result_io.hpp>

#include <filesystem>
#include <memory>
#include <string>

namespace rtr {
namespace {

class AlignResultSaveOp final : public cgraph::MemoryOperator {
 public:
  AlignResultSaveOp() {
    op_id_ = "rtr.align_result.save";
    signature_.inputs["align"] = align_port("align");
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["file"] = align_file_port("file");
    capability_.summary = "Serialize AlignResult to JSON file (AlignResultIO)";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "把图上 AlignResult Value 本地化写出为 RTR AlignResult JSON 文件，"
        "并吐出 align_result_file Artifact。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("align") || !inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.align_result.save: missing align/path");
    }
    const auto path = std::filesystem::path(inputs.at("path").payload.as_string());
    std::filesystem::create_directories(path.parent_path());
    const Ddx::AlignResult align = align_result_from_data(inputs.at("align"));
    if (!Ddx::AlignResultIO::saveAlignResult(path.string(), align)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.align_result.save: saveAlignResult failed");
    }
    return {{"file", file_artifact_from_path(path, "rtr.type.align_result_file",
                                             "rtr.semantic.align_result_file")}};
  }
};

class AlignResultLoadOp final : public cgraph::MemoryOperator {
 public:
  AlignResultLoadOp() {
    op_id_ = "rtr.align_result.load";
    signature_.inputs["path"] = path_port("path");
    signature_.outputs["align"] = align_port("align");
    signature_.outputs["file"] = align_file_port("file");
    capability_.summary = "Load AlignResult JSON file (AlignResultIO)";
    cost_.cost_class = "cpu.light";
    effect_.effect = cgraph::EffectClass::External;
    effect_.cache = cgraph::CachePolicy::Volatile;
    usage_.principle =
        "从 RTR AlignResult JSON 本地文件读入图上 AlignResult Value。";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    if (!inputs.count("path")) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.align_result.load: missing path");
    }
    const auto path = std::filesystem::path(inputs.at("path").payload.as_string());
    if (!std::filesystem::is_regular_file(path)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.align_result.load: file missing");
    }
    Ddx::AlignResult align;
    if (!Ddx::AlignResultIO::loadAlignResult(path.string(), align)) {
      throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed,
                                  "rtr.align_result.load: loadAlignResult failed");
    }
    return {{"align", align_result_to_data(align)},
            {"file", file_artifact_from_path(path, "rtr.type.align_result_file",
                                             "rtr.semantic.align_result_file")}};
  }
};

}  // namespace

void register_align_result_io(cgraph::OpRegistry& registry) {
  registry.add(std::make_shared<AlignResultSaveOp>());
  registry.add(std::make_shared<AlignResultLoadOp>());
}

}  // namespace rtr
