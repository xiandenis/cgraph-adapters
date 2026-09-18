#include "cgraph/ops.hpp"
#include "fx_typed.hpp"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace fixture {
namespace {

class SubOp final : public cgraph::MemoryOperator {
 public:
  SubOp() {
    op_id_ = "fx.sub";
    signature_.inputs["a"] = fx_typed::float_port("a");
    signature_.inputs["b"] = fx_typed::float_port("b");
    signature_.outputs["diff"] = fx_typed::float_port("diff");
    capability_.summary = "Subtract two floats (a - b)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire floats into a and b; read diff.";
    usage_.tune = "No parameters.";
    usage_.inspect = "diff = a - b.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const double a = fx_typed::require_float_port(data_in, "a", "fx.sub");
    const double b = fx_typed::require_float_port(data_in, "b", "fx.sub");
    return {{"diff", fx_typed::make_number_float(a - b)}};
  }
};

class DivOp final : public cgraph::MemoryOperator {
 public:
  DivOp() {
    op_id_ = "fx.div";
    signature_.inputs["a"] = fx_typed::float_port("a");
    signature_.inputs["b"] = fx_typed::float_port("b");
    signature_.outputs["quot"] = fx_typed::float_port("quot");
    capability_.summary = "Divide two floats";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire floats into a and b; read quot.";
    usage_.tune = "Division by zero fails.";
    usage_.inspect = "quot = a / b.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const double a = fx_typed::require_float_port(data_in, "a", "fx.div");
    const double b = fx_typed::require_float_port(data_in, "b", "fx.div");
    if (b == 0.0) {
      throw std::invalid_argument("fx.div: division by zero");
    }
    return {{"quot", fx_typed::make_number_float(a / b)}};
  }
};

class NegOp final : public cgraph::MemoryOperator {
 public:
  NegOp() {
    op_id_ = "fx.neg";
    signature_.inputs["in"] = fx_typed::float_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    capability_.summary = "Negate a float (-x)";
    capability_.tags = {"math", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a float into in; read out.";
    usage_.tune = "No parameters.";
    usage_.inspect = "out = -in.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json&, const cgraph::ExecContext&) const override {
    const double x = fx_typed::require_float_port(data_in, "in", "fx.neg");
    return {{"out", fx_typed::make_number_float(-x)}};
  }
};

class PrintOp final : public cgraph::MemoryOperator {
 public:
  PrintOp() {
    op_id_ = "fx.print";
    signature_.inputs["in"] = fx_typed::float_port("in");
    signature_.outputs["out"] = fx_typed::float_port("out");
    cgraph::ParamSpec label;
    label.name = "label";
    label.dtype = "string";
    label.default_value = "print";
    label.doc = "Prefix printed to stdout";
    label.bindable = false;
    signature_.params["label"] = std::move(label);
    capability_.summary = "Print float to stdout and pass through";
    capability_.tags = {"print", "debug", "fixture"};
    cost_.cost_class = "cpu.tiny";
    usage_.connect = "Wire a float into in; out equals in.";
    usage_.tune = "params.label prefixes the line.";
    usage_.inspect = "Writes one line to stdout; does not change the value.";
  }

  std::map<std::string, cgraph::DataObject> execute(
      const std::map<std::string, cgraph::DataObject>& data_in,
      const nlohmann::json& params, const cgraph::ExecContext&) const override {
    const double x = fx_typed::require_float_port(data_in, "in", "fx.print");
    std::string label = "print";
    if (params.is_object() && params.contains("label") &&
        params["label"].is_string()) {
      label = params["label"].get<std::string>();
    }
    std::cout << "[" << label << "] " << x << std::endl;
    return {{"out", fx_typed::make_number_float(x)}};
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
