#include "cgraph/ops.hpp"
#include "fx_data.hpp"
#include "cgraph/validate.hpp"

#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class ConstrainedOp final : public cgraph::MemoryOperator {
 public:
  ConstrainedOp() {
    op_id_ = "fx.constrained";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    cgraph::ParamSpec a;
    a.name = "a";
    a.dtype = "int";
    a.default_value = 1;
    a.invalidate = true;
    signature_.params["a"] = std::move(a);
    cgraph::ParamSpec b;
    b.name = "b";
    b.dtype = "int";
    b.default_value = 1;
    b.invalidate = true;
    signature_.params["b"] = std::move(b);
    capability_.summary = "Fixture cross-param constraint";
    cost_.cost_class = "cpu.tiny";
  }

  void validate_instance(const cgraph::ValidationContext& ctx,
                         cgraph::ValidationReport& out) const override {
    int a = 0;
    int b = 0;
    if (ctx.node.params.is_object()) {
      if (ctx.node.params.contains("a") && ctx.node.params["a"].is_number_integer()) {
        a = ctx.node.params["a"].get<int>();
      }
      if (ctx.node.params.contains("b") && ctx.node.params["b"].is_number_integer()) {
        b = ctx.node.params["b"].get<int>();
      }
    }
    if (a + b <= 0) {
      out.add("fixture.bad_combo", "a+b must be > 0", ctx.node.id);
    }
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    return fx::wrap(signature_, {{"out", params}});
  }
};

class BadSeedDeclOp final : public cgraph::MemoryOperator {
 public:
  BadSeedDeclOp() {
    op_id_ = "fx.bad_seed_decl";
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, cgraph::type_ids::tensor(), cgraph::SemanticSpec::of("number"));
    // Intentionally no seed ParamSpec while EffectSpec requires seed_param.
    effect_.effect = cgraph::EffectClass::Stochastic;
    effect_.cache = cgraph::CachePolicy::Memoizable;
    effect_.seed_param = "seed";
    capability_.summary = "Malformed EffectSpec for gate tests";
    cost_.cost_class = "cpu.tiny";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>&,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    return fx::wrap(signature_, {{"out", 0}});
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_constrained() {
  return std::make_shared<ConstrainedOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_bad_seed_decl() {
  return std::make_shared<BadSeedDeclOp>();
}

}  // namespace fixture
