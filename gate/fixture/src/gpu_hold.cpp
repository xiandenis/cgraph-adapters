#include "cgraph/ops.hpp"
#include "fx_data.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

namespace fixture {
namespace {

class GpuHoldOp final : public cgraph::MemoryOperator {
 public:
  GpuHoldOp() {
    op_id_ = "fx.gpu_hold";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    cgraph::ParamSpec ms;
    ms.name = "ms";
    ms.dtype = "int";
    ms.default_value = 200;
    ms.invalidate = false;
    ms.bindable = false;
    ms.doc = "Sleep duration in execute (fixture gate test)";
    signature_.params["ms"] = std::move(ms);
    capability_.summary = "Fixture: hold GPU slot while sleeping";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire json in; out copies in after sleep.";
    usage_.tune = "params.ms controls sleep (default 200).";
    usage_.inspect = "Slot gpu.lock exists while running.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext& ctx) const override {
    const auto inputs = fx::unwrap(data_in);
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.gpu_hold: missing input 'in'");
    }
    int ms = 200;
    if (params.is_object() && params.contains("ms") && params["ms"].is_number_integer()) {
      ms = params["ms"].get<int>();
    }
    if (ms < 0) {
      ms = 0;
    }
    const std::filesystem::path lock_path = ctx.workdir / "gpu.lock";
    {
      std::ofstream lock(lock_path, std::ios::trunc);
      lock << "running\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    std::error_code ec;
    std::filesystem::remove(lock_path, ec);
    return fx::wrap(signature_, {{"out", it->second}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_gpu_hold() {
  return std::make_shared<GpuHoldOp>();
}

}  // namespace fixture
