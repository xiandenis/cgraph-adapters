#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class AddOp final : public cgraph::MemoryOperator {
 public:
  AddOp() {
    op_id_ = "fx.add";
    signature_.inputs["a"] = fx_typed::float_port("a");
    signature_.inputs["b"] = fx_typed::float_port("b");
    signature_.outputs["sum"] = fx_typed::float_port("sum");
    capability_.summary = "Add two floats";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire floats into a and b; read sum.";
    usage_.tune = "No parameters.";
    usage_.inspect = "sum = a + b.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const double a = fx_typed::require_float_port(data_in, "a", "fx.add");
    const double b = fx_typed::require_float_port(data_in, "b", "fx.add");
    return {{"sum", fx_typed::make_number_float(a + b)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_add() {
  return std::make_shared<AddOp>();
}

}  // namespace fixture
