#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

double param_or_input_exp(
    const std::map<std::string, cgraph::DataObject>& data_in,
    const nlohmann::json& params) {
  const auto it = data_in.find("exp");
  if (it != data_in.end()) {
    return fx_typed::require_float(it->second, "fx.pow", "exp");
  }
  if (params.is_object() && params.contains("exp") && params["exp"].is_number()) {
    return params["exp"].get<double>();
  }
  throw std::invalid_argument("fx.pow: missing exp (input or params.exp)");
}

class PowOp final : public cgraph::MemoryOperator {
 public:
  PowOp() {
    op_id_ = "fx.pow";
    signature_.inputs["base"] = fx_typed::float_port("base");
    auto exp_in = fx_typed::float_port("exp");
    exp_in.optional = true;
    signature_.inputs["exp"] = std::move(exp_in);
    signature_.outputs["out"] = fx_typed::float_port("out");
    cgraph::ParamSpec exp;
    exp.name = "exp";
    exp.dtype = "number";
    exp.doc = "Optional exponent when exp input is unwired";
    exp.bindable = true;
    signature_.params["exp"] = std::move(exp);
    capability_.summary = "Power: out = base ** exp";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire float base; wire exp or set params.exp; read out.";
    usage_.tune = "params.exp used only if exp input absent.";
    usage_.inspect = "Domain errors throw with fx.pow prefix.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const double base = fx_typed::require_float_port(data_in, "base", "fx.pow");
    const double exp = param_or_input_exp(data_in, params);
    const double out = std::pow(base, exp);
    if (!std::isfinite(out)) {
      throw std::invalid_argument("fx.pow: non-finite result");
    }
    return {{"out", fx_typed::make_number_float(out)}};
  }
};

class UnaryMathOp final : public cgraph::MemoryOperator {
 public:
  using Fn = double (*)(double);

  UnaryMathOp(const char* id, const char* summary, Fn fn, bool reject_nonpositive,
              bool reject_negative)
      : fn_(fn),
        reject_nonpositive_(reject_nonpositive),
        reject_negative_(reject_negative) {
    op_id_ = id;
    signature_.inputs["in"] = fx_typed::float_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = summary;
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a float into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Domain errors throw with op_id prefix.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const double x = fx_typed::require_float_port(data_in, "in", op_id_);
    if (reject_nonpositive_ && !(x > 0.0)) {
      throw std::invalid_argument(op_id_ + ": domain requires in > 0");
    }
    if (reject_negative_ && x < 0.0) {
      throw std::invalid_argument(op_id_ + ": domain requires in >= 0");
    }
    const double out = fn_(x);
    if (!std::isfinite(out)) {
      throw std::invalid_argument(op_id_ + ": non-finite result");
    }
    return {{"out", fx_typed::make_number_float(out)}};
  }

 private:
  Fn fn_;
  bool reject_nonpositive_;
  bool reject_negative_;
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_pow() {
  return std::make_shared<PowOp>();
}
std::shared_ptr<cgraph::MemoryOperator> make_log() {
  return std::make_shared<UnaryMathOp>("fx.log", "Natural logarithm",
                                      static_cast<UnaryMathOp::Fn>(std::log), true,
                                      false);
}
std::shared_ptr<cgraph::MemoryOperator> make_log2() {
  return std::make_shared<UnaryMathOp>("fx.log2", "Base-2 logarithm",
                                      static_cast<UnaryMathOp::Fn>(std::log2), true,
                                      false);
}
std::shared_ptr<cgraph::MemoryOperator> make_exp() {
  return std::make_shared<UnaryMathOp>("fx.exp", "Exponential e^x",
                                      static_cast<UnaryMathOp::Fn>(std::exp), false,
                                      false);
}
std::shared_ptr<cgraph::MemoryOperator> make_sin() {
  return std::make_shared<UnaryMathOp>("fx.sin", "Sine (radians)",
                                      static_cast<UnaryMathOp::Fn>(std::sin), false,
                                      false);
}
std::shared_ptr<cgraph::MemoryOperator> make_cos() {
  return std::make_shared<UnaryMathOp>("fx.cos", "Cosine (radians)",
                                      static_cast<UnaryMathOp::Fn>(std::cos), false,
                                      false);
}
std::shared_ptr<cgraph::MemoryOperator> make_tan() {
  return std::make_shared<UnaryMathOp>("fx.tan", "Tangent (radians)",
                                      static_cast<UnaryMathOp::Fn>(std::tan), false,
                                      false);
}
std::shared_ptr<cgraph::MemoryOperator> make_sqrt() {
  return std::make_shared<UnaryMathOp>("fx.sqrt", "Square root",
                                      static_cast<UnaryMathOp::Fn>(std::sqrt), false,
                                      true);
}
std::shared_ptr<cgraph::MemoryOperator> make_tanh() {
  return std::make_shared<UnaryMathOp>("fx.tanh", "Hyperbolic tangent",
                                      static_cast<UnaryMathOp::Fn>(std::tanh), false,
                                      false);
}

}  // namespace fixture
