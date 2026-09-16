#include "cgraph/ops.hpp"
#include "cgraph/sandbox.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class StampProcOp final : public cgraph::ProcessOperator {
 public:
  StampProcOp() {
    op_id_ = "fx.stamp_proc";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, "file");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    cgraph::ParamSpec tag;
    tag.name = "tag";
    tag.dtype = "string";
    tag.invalidate = true;
    tag.doc = "Identity tag wrapped into out.t";
    tag.bindable = false;
    signature_.params["tag"] = std::move(tag);
    capability_.summary = "Process isomorphic stamp: read JSON file, write {p,t}";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a json file into in; read file out.";
    usage_.tune = "Set params.tag (string, invalidate).";
    usage_.inspect = "out file bytes are {\"p\": <in json>, \"t\": tag}.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>&,
      const nlohmann::json& params, const cgraph::ExecContext& ctx) const override {
    if (ctx.sandbox == nullptr) {
      throw std::invalid_argument("fx.stamp_proc: sandbox required");
    }
    if (!params.is_object() || !params.contains("tag") || !params["tag"].is_string()) {
      throw std::invalid_argument("fx.stamp_proc: missing or invalid param 'tag'");
    }
    const std::string text = ctx.sandbox->read_file(ctx.sandbox->input_path("in"));
    const nlohmann::json in = nlohmann::json::parse(text);
    nlohmann::json out = nlohmann::json::object();
    out["p"] = in;
    out["t"] = params["tag"];
    ctx.sandbox->write_file(ctx.sandbox->output_path("out"), out.dump());
    return {{"out", ctx.sandbox->output_artifact("out", ctx.artifact_digest_mode)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_stamp_proc() {
  return std::make_shared<StampProcOp>();
}

}  // namespace fixture
