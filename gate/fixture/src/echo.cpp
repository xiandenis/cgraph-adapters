#include "cgraph/ops.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class EchoOp final : public cgraph::MemoryOperator {
 public:
  EchoOp() {
    op_id_ = "fx.echo";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Copy a JSON value";
    capability_.tags = {"math", "fixture", "io"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire any json value into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out is identical to in.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.echo: missing input 'in'");
    }
    return {{"out", it->second}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_echo() {
  return std::make_shared<EchoOp>();
}

}  // namespace fixture
