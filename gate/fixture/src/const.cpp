#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>

namespace fixture {
namespace {

class ConstOp final : public cgraph::MemoryOperator {
 public:
  ConstOp() {
    op_id_ = "fx.const";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "json";
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit a JSON literal";
    capability_.tags = {"math", "fixture", "io"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Connect out to a json value port.";
    usage_.tune = "Set params.value to the literal.";
    usage_.inspect = "out equals params.value byte-for-byte after canon.";
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
