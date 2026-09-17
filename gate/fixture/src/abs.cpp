#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class AbsOp final : public cgraph::MemoryOperator {
 public:
  AbsOp() {
    op_id_ = "fx.abs";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    capability_.summary = "Absolute value of a JSON number";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a JSON number into in; read out.";
    usage_.tune = "Integer uses int64 path (cutoff digest); else |double|.";
    usage_.inspect = "out = |in|. abs(5) and abs(-5) share output_digest.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.abs: missing input 'in'");
    }
    if (!it->second.is_number()) {
      throw std::invalid_argument("fx.abs: input must be a JSON number");
    }
    if (it->second.is_number_integer()) {
      const auto v = it->second.get<std::int64_t>();
      if (v == std::numeric_limits<std::int64_t>::min()) {
        throw std::invalid_argument("fx.abs: int64 overflow");
      }
      const std::int64_t out = v < 0 ? -v : v;
      return fx::wrap(signature_, {{"out", out}});
    }
    return fx::wrap(signature_, {{"out", std::fabs(it->second.get<double>())}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_abs() {
  return std::make_shared<AbsOp>();
}

}  // namespace fixture
