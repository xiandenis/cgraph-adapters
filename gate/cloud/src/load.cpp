#include "cgraph/artifact.hpp"
#include "cgraph/ops.hpp"
#include "cloud/cloud_value.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace cloud {
namespace {

class LoadOp final : public cgraph::MemoryOperator {
 public:
  LoadOp() {
    op_id_ = "cloud.load";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, "file");
    signature_.inputs["in"].formats = {"ply", "pcd"};
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "cloud");
    signature_.outputs["out"].formats = {"native"};
    signature_.outputs["out"].schema = {{"fields", nlohmann::json::array({"xyz"})}};
    cgraph::ParamSpec format;
    format.name = "format";
    format.dtype = "string";
    format.default_value = "";
    format.bindable = false;
    signature_.params["format"] = std::move(format);
    capability_.summary = "Load ASCII PLY/PCD into value/cloud";
    cost_.cost_class = "cpu.io";
    usage_.connect = "Wire a ply/pcd file into in; read cloud from out.";
    usage_.tune = "params.format optional; empty uses path suffix.";
    usage_.inspect = "out is JSON SoA with fields=[xyz].";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json& params,
      const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("cloud.load: missing input 'in'");
    }
    const cgraph::Artifact src = cgraph::artifact_from_json(it->second);
    std::string format;
    if (params.is_object()) {
      format = params.value("format", std::string());
    }
    if (format.empty()) {
      const auto ext = src.path.extension().string();
      if (ext == ".ply" || ext == ".PLY") {
        format = "ply";
      } else if (ext == ".pcd" || ext == ".PCD") {
        format = "pcd";
      }
    }
    nlohmann::json cloud;
    if (format == "ply") {
      cloud = load_ply_ascii(src.path);
    } else if (format == "pcd") {
      cloud = load_pcd_ascii(src.path);
    } else {
      throw std::invalid_argument("cloud.load: unsupported format '" + format + "'");
    }
    return {{"out", std::move(cloud)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cloud_load() {
  return std::make_shared<LoadOp>();
}

}  // namespace cloud
