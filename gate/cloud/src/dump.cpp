#include "cgraph/artifact.hpp"
#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace cloud {
namespace {

class DumpOp final : public cgraph::MemoryOperator {
 public:
  DumpOp() {
    op_id_ = "cloud.dump";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "cloud");
    signature_.inputs["in"].formats = {"native"};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    signature_.outputs["out"].formats = {"ply", "pcd"};
    cgraph::ParamSpec format;
    format.name = "format";
    format.dtype = "string";
    format.bindable = false;
    signature_.params["format"] = std::move(format);
    capability_.summary = "Dump value/cloud to ASCII PLY/PCD";
    cost_.cost_class = "cpu.io";
    usage_.connect = "Wire cloud into in; read file handle from out.";
    usage_.tune = "params.format must be ply or pcd.";
    usage_.inspect = "Artifact path under workdir/outputs/out.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext& ctx) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.dump: missing input 'in'");
    }
    if (ctx.workdir.empty()) {
      throw std::invalid_argument("cloud.dump: workdir required");
    }
    std::string format;
    if (params.is_object()) {
      format = params.value("format", std::string());
    }
    if (format != "ply" && format != "pcd") {
      throw std::invalid_argument("cloud.dump: format must be ply or pcd");
    }
    const std::filesystem::path out = ctx.workdir / "outputs" / "out";
    std::filesystem::create_directories(out.parent_path());
    if (format == "ply") {
      dump_ply_ascii(it->second, out);
    } else {
      dump_pcd_ascii(it->second, out);
    }
    return {{"out", cgraph::artifact_to_json(cgraph::make_file_artifact(out))}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_dump() {
  return std::make_shared<DumpOp>();
}

}  // namespace cloud
