#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class EchoOp final : public cgraph::MemoryOperator {
 public:
  EchoOp() {
    op_id_ = "fx.echo";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    capability_.summary = "Copy a JSON value";
    capability_.tags = {"math", "fixture", "io"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire any json value into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out is identical to in.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.echo: missing input 'in'");
    }
    return fx::wrap(signature_, {{"out", it->second}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_echo() {
  return std::make_shared<EchoOp>();
}

}  // namespace fixture
