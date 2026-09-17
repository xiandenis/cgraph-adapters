#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <map>
#include <memory>

namespace fixture {
namespace {

class BadCostOp final : public cgraph::MemoryOperator {
 public:
  BadCostOp() {
    op_id_ = "fx.bad_cost";
    signature_.inputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, cgraph::type_ids::integer(),
                          cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["y"] =
        cgraph::make_port("y", cgraph::PortKind::Value, cgraph::type_ids::integer(),
                          cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Fixture op with invalid CostHint YAML (Step 20 gate)";
    cost_.cost_class = "cpu.tiny";
    load_cost_description("fixture");
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& /*params*/, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("x");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.bad_cost: missing input 'x'");
    }
    return fx::wrap(signature_, {{"y", it->second}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_bad_cost() {
  return std::make_shared<BadCostOp>();
}

}  // namespace fixture
