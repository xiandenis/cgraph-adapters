#include "cgraph/ops.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace fixture {
namespace {

double require_number(const std::map<std::string, nlohmann::json>& inputs,
                      const std::string& name, const char* op) {
  const auto it = inputs.find(name);
  if (it == inputs.end()) {
    throw std::invalid_argument(std::string(op) + ": missing input '" + name + "'");
  }
  if (!it->second.is_number()) {
    throw std::invalid_argument(std::string(op) + ": input '" + name +
                                "' must be a JSON number");
  }
  return it->second.get<double>();
}

double param_or_input_exp(const std::map<std::string, nlohmann::json>& inputs,
                          const nlohmann::json& params) {
  // Wired input wins; Runtime.resolve_params also mirrors it into params.exp.
  const auto it = inputs.find("exp");
  if (it != inputs.end()) {
    if (!it->second.is_number()) {
      throw std::invalid_argument("fx.pow: input 'exp' must be a JSON number");
    }
    return it->second.get<double>();
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
    signature_.inputs["base"] =
        cgraph::make_port("base", cgraph::PortKind::Value, "json");
    auto exp_in = cgraph::make_port("exp", cgraph::PortKind::Value, "json");
    exp_in.optional = true;
    signature_.inputs["exp"] = std::move(exp_in);
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec exp;
    exp.name = "exp";
    exp.dtype = "number";
    exp.doc = "Optional exponent when exp input is unwired";
    exp.bindable = true;
    signature_.params["exp"] = std::move(exp);
    capability_.summary = "Power: out = base ** exp";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire base; wire exp or set params.exp; read out.";
    usage_.tune = "params.exp used only if exp input absent.";
    usage_.inspect = "Domain errors (e.g. negative**non-int) throw.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const double base = require_number(inputs, "base", "fx.pow");
    const double exp = param_or_input_exp(inputs, params);
    const double out = std::pow(base, exp);
    if (!std::isfinite(out)) {
      throw std::invalid_argument("fx.pow: non-finite result");
    }
    return {{"out", out}};
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
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = summary;
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a JSON number into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "Domain errors throw with op_id prefix.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs, const nlohmann::json&,
      const cgraph::ExecContext&) const override {
    const double x = require_number(inputs, "in", op_id_.c_str());
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
    return {{"out", out}};
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
