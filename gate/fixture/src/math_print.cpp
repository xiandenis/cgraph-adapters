#include "cgraph/ops.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

bool sub_would_overflow(std::int64_t a, std::int64_t b) {
  const auto max = std::numeric_limits<std::int64_t>::max();
  const auto min = std::numeric_limits<std::int64_t>::min();
  return (b > 0 && a < min + b) || (b < 0 && a > max + b);
}

class SubOp final : public cgraph::MemoryOperator {
 public:
  SubOp() {
    op_id_ = "fx.sub";
    signature_.inputs["a"] =
        cgraph::make_port("a", cgraph::PortKind::Value, "json");
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, "json");
    signature_.outputs["diff"] =
        cgraph::make_port("diff", cgraph::PortKind::Value, "json");
    capability_.summary = "Subtract two JSON numbers (a - b)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire JSON numbers into a and b; read diff.";
    usage_.tune = "Both integers → int64; else double.";
    usage_.inspect = "diff = a - b.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto ait = inputs.find("a");
    const auto bit = inputs.find("b");
    if (ait == inputs.end() || bit == inputs.end()) {
      throw std::invalid_argument("fx.sub: missing input 'a' or 'b'");
    }
    if (!ait->second.is_number() || !bit->second.is_number()) {
      throw std::invalid_argument("fx.sub: inputs must be JSON numbers");
    }
    if (ait->second.is_number_integer() && bit->second.is_number_integer()) {
      const auto a = ait->second.get<std::int64_t>();
      const auto b = bit->second.get<std::int64_t>();
      if (sub_would_overflow(a, b)) {
        throw std::invalid_argument("fx.sub: int64 overflow");
      }
      return {{"diff", a - b}};
    }
    return {{"diff", ait->second.get<double>() - bit->second.get<double>()}};
  }
};

class DivOp final : public cgraph::MemoryOperator {
 public:
  DivOp() {
    op_id_ = "fx.div";
    signature_.inputs["a"] =
        cgraph::make_port("a", cgraph::PortKind::Value, "json");
    signature_.inputs["b"] =
        cgraph::make_port("b", cgraph::PortKind::Value, "json");
    signature_.outputs["quot"] =
        cgraph::make_port("quot", cgraph::PortKind::Value, "json");
    capability_.summary = "Divide two JSON numbers (float quot)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire numbers into a and b; read quot.";
    usage_.tune = "No parameters. Division by zero fails.";
    usage_.inspect = "quot = double(a) / double(b).";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto ait = inputs.find("a");
    const auto bit = inputs.find("b");
    if (ait == inputs.end() || bit == inputs.end()) {
      throw std::invalid_argument("fx.div: missing input 'a' or 'b'");
    }
    if (!ait->second.is_number() || !bit->second.is_number()) {
      throw std::invalid_argument("fx.div: inputs must be JSON numbers");
    }
    const double a = ait->second.get<double>();
    const double b = bit->second.get<double>();
    if (b == 0.0) {
      throw std::invalid_argument("fx.div: division by zero");
    }
    return {{"quot", a / b}};
  }
};

class NegOp final : public cgraph::MemoryOperator {
 public:
  NegOp() {
    op_id_ = "fx.neg";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    capability_.summary = "Negate a JSON number (-x)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a JSON number into in; read out.";
    usage_.tune = "Integer uses int64 path; else double.";
    usage_.inspect = "out = -in.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.neg: missing input 'in'");
    }
    if (!it->second.is_number()) {
      throw std::invalid_argument("fx.neg: input must be a JSON number");
    }
    if (it->second.is_number_integer()) {
      const auto v = it->second.get<std::int64_t>();
      if (v == std::numeric_limits<std::int64_t>::min()) {
        throw std::invalid_argument("fx.neg: int64 overflow");
      }
      return {{"out", -v}};
    }
    return {{"out", -it->second.get<double>()}};
  }
};

class PrintOp final : public cgraph::MemoryOperator {
 public:
  PrintOp() {
    op_id_ = "fx.print";
    signature_.inputs["in"] =
        cgraph::make_port("in", cgraph::PortKind::Value, "json");
    signature_.outputs["out"] =
        cgraph::make_port("out", cgraph::PortKind::Value, "json");
    cgraph::ParamSpec label;
    label.name = "label";
    label.dtype = "string";
    label.default_value = "print";
    label.doc = "Prefix printed to stdout";
    label.bindable = false;
    signature_.params["label"] = std::move(label);
    capability_.summary = "Print JSON to stdout and pass through";
    capability_.tags = {"print", "debug", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire any json into in; out equals in.";
    usage_.tune = "params.label prefixes the line.";
    usage_.inspect = "Writes one line to stdout; does not change the value.";
  }

  std::map<std::string, nlohmann::json> execute(
      const std::map<std::string, nlohmann::json>& inputs,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const auto it = inputs.find("in");
    if (it == inputs.end()) {
      throw std::invalid_argument("fx.print: missing input 'in'");
    }
    std::string label = "print";
    if (params.is_object() && params.contains("label") &&
        params["label"].is_string()) {
      label = params["label"].get<std::string>();
    }
    std::cout << "[" << label << "] " << it->second.dump() << std::endl;
    return {{"out", it->second}};
  }
};

}  // namespace

std::shared_ptr<cgraph::MemoryOperator> make_sub() {
  return std::make_shared<SubOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_div() {
  return std::make_shared<DivOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_neg() {
  return std::make_shared<NegOp>();
}

std::shared_ptr<cgraph::MemoryOperator> make_print() {
  return std::make_shared<PrintOp>();
}

}  // namespace fixture
