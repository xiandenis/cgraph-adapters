#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class ConstScalarOp final : public cgraph::MemoryOperator {
 public:
  ConstScalarOp() {
    op_id_ = "fx.const_scalar";
    signature_.outputs["out"] = fx_typed::float_port("out");
    cgraph::ParamSpec value;
    value.name = "value";
    value.dtype = "number";
    value.bindable = false;
    signature_.params["value"] = std::move(value);
    capability_.summary = "Emit a float scalar";
    capability_.tags = {"math", "fixture", "io"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs. Wire out to float math ports.";
    usage_.tune = "params.value must be a number.";
    usage_.inspect = "out is cgraph.type.float + number.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    if (!params.is_object() || !params.contains("value") ||
        !params["value"].is_number()) {
      throw std::invalid_argument("fx.const_scalar: params.value number required");
    }
    return {{"out", fx_typed::make_number_float(params["value"].get<double>())}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_const_scalar() {
  return std::make_shared<ConstScalarOp>();
}

}  // namespace fixture
