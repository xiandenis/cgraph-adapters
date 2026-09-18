#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class AbsOp final : public cgraph::MemoryOperator {
 public:
  AbsOp() {
    op_id_ = "fx.abs";
    signature_.inputs["in"] = fx_typed::float_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Absolute value of a float";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a float into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out = |in|.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const double x = fx_typed::require_float_port(data_in, "in", "fx.abs");
    return {{"out", fx_typed::make_number_float(std::fabs(x))}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_abs() {
  return std::make_shared<AbsOp>();
}

}  // namespace fixture
