#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

bool add_would_overflow(std::int64_t a, std::int64_t b) {
  const auto max = std::numeric_limits<std::int64_t>::max();
  const auto min = std::numeric_limits<std::int64_t>::min();
  return (b > 0 && a > max - b) || (b < 0 && a < min - b);
}

class AddOp final : public cgraph::MemoryOperator {
 public:
  AddOp() {
    op_id_ = "fx.add";
    signature_.inputs["a"] =
        cgraph::make_port("a", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    signature_.outputs["sum"] =
        cgraph::make_port("sum", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("cgraph.semantic.number"));
    capability_.summary = "Add two JSON numbers (int64 path or float)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire JSON numbers into a and b; read sum.";
    usage_.tune = "Both integers → int64 (+overflow); else double.";
    usage_.inspect = "sum = a + b.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto ait = inputs.find("a");
    const auto bit = inputs.find("b");
    if (ait == inputs.end() || bit == inputs.end()) {
      throw std::invalid_argument("fx.add: missing input 'a' or 'b'");
    }
    if (!ait->second.is_number() || !bit->second.is_number()) {
      throw std::invalid_argument("fx.add: inputs must be JSON numbers");
    }
    if (ait->second.is_number_integer() && bit->second.is_number_integer()) {
      const auto a = ait->second.get<std::int64_t>();
      const auto b = bit->second.get<std::int64_t>();
      if (add_would_overflow(a, b)) {
        throw std::invalid_argument("fx.add: int64 overflow");
      }
      return fx::wrap(signature_, {{"sum", a + b}});
    }
    return fx::wrap(signature_, {{"sum", ait->second.get<double>() + bit->second.get<double>()}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_add() {
  return std::make_shared<AddOp>();
}

}  // namespace fixture
