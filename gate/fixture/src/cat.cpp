#include "cgraph/ops.hpp"
#include "fx_data.hpp"
#include "cgraph/sandbox.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class CatOp final : public cgraph::ProcessOperator {
 public:
  CatOp() {
    op_id_ = "fx.cat";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Artifact, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("cgraph.semantic.file"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Artifact, cgraph::type_ids::untyped(),
                          cgraph::SemanticSpec::of("cgraph.semantic.file"));
    capability_.summary = "Byte-copy a file into this node's DiskStore slot";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a file into in; read file out.";
    usage_.tune = "impl_fingerprint changes node_key (case 17).";
    usage_.inspect = "out is a byte copy of in under this slot outputs/out.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json&, const cgraph::ExecContext& ctx) const override {
    if (ctx.sandbox == nullptr) {
      throw std::invalid_argument("fx.cat: sandbox required");
    }
    const std::string bytes = ctx.sandbox->read_file(ctx.sandbox->input_path("in"));
    ctx.sandbox->write_file(ctx.sandbox->output_path("out"), bytes);
    return fx::wrap(signature_, {{"out", ctx.sandbox->output_artifact("out", ctx.artifact_digest_mode)}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_cat() {
  return std::make_shared<CatOp>();
}

}  // namespace fixture
