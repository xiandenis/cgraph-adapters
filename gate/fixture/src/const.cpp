#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>

namespace fixture {
namespace {

class ConstOp final : public cgraph::MemoryOperator {
 public:
  ConstOp() {
    op_id_ = "fx.const";
    signature_.outputs["out"] = cgraph::make_port(
        "out", cgraph::PortKind::Value, cgraph::type_ids::untyped(),
        cgraph::SemanticSpec::of("cgraph.semantic.document"));
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "string";
    value.bindable = false;
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit an untyped document literal (not for math ports)";
    capability_.tags = {"debug", "fixture", "io"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Do not wire out into fx math float/vector/matrix ports.";
    usage_.tune = "Set params.value to a JSON literal.";
    usage_.inspect = "For math graphs use fx.const_scalar / const_vector / const_matrix.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    nlohmann::json value = nullptr;
    if (params.is_object() && params.contains("value")) {
      value = params["value"];
    }
    return fx::wrap(signature_, {{"out", std::move(value)}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_const() {
  return std::make_shared<ConstOp>();
}

}  // namespace fixture
