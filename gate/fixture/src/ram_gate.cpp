#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>

namespace fixture {
namespace {

class RamGateOp final : public cgraph::MemoryOperator {
 public:
  RamGateOp() {
    op_id_ = "fx.ram_gate";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Fixture: RAM gate target (echo)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json in; out copies in.";
    usage_.tune = "Set node resources.ram_mb for defer tests.";
    usage_.inspect = "Identity passthrough.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.ram_gate: missing input 'in'");
    }
    return fx::wrap(signature_, {{"out", it->second}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_ram_gate() {
  return std::make_shared<RamGateOp>();
}

}  // namespace fixture
