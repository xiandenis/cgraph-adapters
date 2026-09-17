#include "cgraph/cost.hpp"
#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class ScaleOp final : public cgraph::MemoryOperator {
 public:
  ScaleOp() {
    op_id_ = "fx.scale";
    signature_.inputs["x"] =
        cgraph::make_port("x", cgraph::PortKind::Value, "int");
    signature_.outputs["y"] =
        cgraph::make_port("y", cgraph::PortKind::Value, "int");
    cgraph::ParamSpec k;
    k.name = "k";
    k.dtype = "int";
    k.domain.min = 0;
    k.domain.exclusive_min = true;
    k.doc = "k > 0";
    signature_.params["k"] = std::move(k);
    capability_.summary = "Multiply an integer by k (k > 0)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Connect int into x; read y.";
    usage_.tune = "k must be > 0. k<=0 is rejected before execute.";
    usage_.inspect = "y = x * k.";
    cgraph::apply_legacy_cost_class(cost_);
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto xit = inputs.find("x");
    if (xit == inputs.end()) {
      throw std::invalid_argument("fx.scale: missing input 'x'");
    }
    if (!params.is_object() || !params.contains("k")) {
      throw std::invalid_argument("fx.scale: missing param 'k'");
    }
    const auto x = xit->second.get<std::int64_t>();
    const auto k = params["k"].get<std::int64_t>();
    return fx::wrap(signature_, {{"y", x * k}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_scale() {
  return std::make_shared<ScaleOp>();
}

}  // namespace fixture
