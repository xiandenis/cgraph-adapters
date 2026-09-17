#include "cgraph/ops.hpp"
#include "fx_data.hpp"
#include "cgraph/sandbox.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class SneakOp final : public cgraph::ProcessOperator {
 public:
  SneakOp() {
    op_id_ = "fx.sneak";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, "file");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, "file");
    cgraph::ParamSpec undeclared;
    undeclared.name = "undeclared";
    undeclared.dtype = "string";
    undeclared.invalidate = true;
    undeclared.doc = "Absolute path to read illegally; empty skips the read";
    signature_.params["undeclared"] = std::move(undeclared);
    capability_.summary = "test-only";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Test-only. Do not put on the Studio palette.";
    usage_.tune = "undeclared: path to sneak-read.";
    usage_.inspect = "Writes undeclared_extra; undeclared read must fail the node.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json& params, const cgraph::ExecContext& ctx) const override {
    if (ctx.sandbox == nullptr) {
      throw std::invalid_argument("fx.sneak: sandbox required");
    }
    ctx.sandbox->write_file(ctx.sandbox->root() / "undeclared_extra", "sneak");
    const std::string bytes = ctx.sandbox->read_file(ctx.sandbox->input_path("in"));
    ctx.sandbox->write_file(ctx.sandbox->output_path("out"), bytes);
    std::string sneak_path;
    if (params.is_object() && params.contains("undeclared") &&
        params["undeclared"].is_string()) {
      sneak_path = params["undeclared"].get<std::string>();
    }
    if (!sneak_path.empty()) {
      ctx.sandbox->read_file(sneak_path);
    }
    return fx::wrap(signature_, {{"out", ctx.sandbox->output_artifact("out", ctx.artifact_digest_mode)}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_sneak() {
  return std::make_shared<SneakOp>();
}

}  // namespace fixture
