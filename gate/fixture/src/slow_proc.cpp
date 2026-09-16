#include "cgraph/ops.hpp"

#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

namespace fixture {
namespace {

class SlowProcOp final : public cgraph::ProcessOperator {
 public:
  SlowProcOp() {
    op_id_ = "fx.slow_proc";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Fixture Process: sleep longer than timeout_s";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json in; sleeps 3s in execute.";
    usage_.tune = "Set node resources.timeout_s to fail fast.";
    usage_.inspect = "Fails with timeout when timeout_s < 3.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext& ctx) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.slow_proc: missing input 'in'");
    }
    const std::filesystem::path cancel_path = ctx.workdir / ".cancel";
    for (int i = 0; i < 30; ++i) {
      if (std::filesystem::exists(cancel_path)) {
        throw cgraph::OperatorError(cgraph::ErrorCode::OpFailed, "cancelled");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return {{"out", it->second}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_slow_proc() {
  return std::make_shared<SlowProcOp>();
}

}  // namespace fixture
