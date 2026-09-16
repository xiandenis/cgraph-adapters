#include "cgraph/ops.hpp"

#include <memory>
#include <random>
#include <stdexcept>

namespace fixture {
namespace {

class RandOp final : public cgraph::MemoryOperator {
 public:
  RandOp() {
    op_id_ = "fx.rand";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec seed;
    seed.name = "seed";
    seed.dtype = "int";
    seed.default_value = nullptr;
    seed.invalidate = true;
    signature_.params["seed"] = std::move(seed);
    cgraph::ParamSpec vol;
    vol.name = "volatile";
    vol.dtype = "bool";
    vol.default_value = false;
    vol.invalidate = false;
    vol.bindable = false;
    signature_.params["volatile"] = std::move(vol);
    capability_.summary = "Fixture RNG (EffectSpec gate)";
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "No inputs; out is a float.";
    usage_.tune = "Set params.seed or params.volatile=true.";
    usage_.inspect = "out is a deterministic float when seed is set.";
    effect_.effect = cgraph::EffectClass::Stochastic;
    effect_.cache = cgraph::CachePolicy::Memoizable;
    effect_.seed_param = "seed";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>&,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    unsigned seed = 0;
    bool have_seed = false;
    if (params.is_object() && params.contains("seed") && !params["seed"].is_null()) {
      if (params["seed"].is_number_unsigned()) {
        seed = params["seed"].get<unsigned>();
        have_seed = true;
      } else if (params["seed"].is_number_integer()) {
        seed = static_cast<unsigned>(params["seed"].get<int>());
        have_seed = true;
      } else {
        throw std::invalid_argument("fx.rand: seed must be integer");
      }
    }
    if (!have_seed) {
      seed = static_cast<unsigned>(std::random_device{}());
    }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return {{"out", dist(rng)}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_rand() {
  return std::make_shared<RandOp>();
}

}  // namespace fixture
