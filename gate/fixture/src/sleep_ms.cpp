#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <chrono>
#include <memory>
#include <thread>

namespace fixture {
namespace {

class SleepMsOp final : public cgraph::MemoryOperator {
 public:
  SleepMsOp() {
    op_id_ = "fx.sleep_ms";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    cgraph::ParamSpec ms;
    ms.name = "ms";
    ms.dtype = "int";
    ms.default_value = 50;
    ms.invalidate = false;
    ms.bindable = false;
    ms.doc = "Sleep duration in execute (parallel wave test)";
    signature_.params["ms"] = std::move(ms);
    capability_.summary = "Fixture: sleep then copy input";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json in; out copies in after sleep.";
    usage_.tune = "params.ms controls sleep (default 50).";
    usage_.inspect = "out is identical to in after delay.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.sleep_ms: missing input 'in'");
    }
    int ms = 50;
    if (params.is_object() && params.contains("ms") && params["ms"].is_number_integer()) {
      ms = params["ms"].get<int>();
    }
    if (ms > 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    return fx::wrap(signature_, {{"out", it->second}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_sleep_ms() {
  return std::make_shared<SleepMsOp>();
}

}  // namespace fixture
